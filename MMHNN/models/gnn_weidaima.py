import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool,LayerNorm, global_mean_pool, global_max_pool
import warnings

warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")

# 如果没有mask，应该是118
num_atom_type = 118
num_chirality_tag = 4

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


#简单的注意力机制
def dot_product_attention(Q, K, V, mask=None):
    # Q, K, V shape: (batch_size, seq_len, d_k)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def add_graph_features_to_nodes(h_2, motif_batch, h_sub_2):
    """
    将每个图的特征添加到对应图中的每个节点上

    参数:
    h_2 (torch.Tensor): 节点特征，形状为 (num_nodes, feature_dim)
    motif_batch (torch.Tensor): 每个节点所属图的标识，形状为 (num_nodes,)
    h_sub_2 (torch.Tensor): 每个图的特征，形状为 (num_graphs, feature_dim)

    返回:
    torch.Tensor: 更新后的节点特征，形状为 (num_nodes, feature_dim)
    """
    # 将每个图的特征扩展到每个节点上
    node_features_to_add = h_sub_2[motif_batch]

    # 将扩展后的图特征加到对应的节点特征上
    updated_h_2 = h_2 + node_features_to_add
    
    return updated_h_2

class GINet(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, dropout=0, pool='mean', device=torch.device("cuda:0")):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.dropout = dropout
        self.id = 0
        self.initial_norm = LayerNorm(55)
        #self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        #self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.pad = nn.Parameter(torch.zeros(self.feat_dim, requires_grad=False)).unsqueeze(dim=0).to(device)

        #nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        #nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.lin = nn.Linear(55, 300)
        # List of MLPs
        self.gnns = nn.ModuleList()
        self.net_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))
            self.net_norms.append(LayerNorm(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_dim, self.feat_dim // 2)
        )

    def forward(self, data_1, data_2):
        
        data_1.x = self.initial_norm(data_1.x, data_1.batch)
        data_1.x = self.lin(data_1.x)
        h_1 = data_1.x
        
        data_2.x = self.initial_norm(data_2.x, data_2.batch)
        data_2.x = self.lin(data_2.x)
        h_2 = data_2.x
        
        for layer in range(self.num_layer):
            h_1 = self.gnns[layer](h_1, data_1.edge_index, data_1.edge_attr)
            h_1 = F.elu(self.net_norms[layer](h_1, data_1.batch))
            
            h_2 = self.gnns[layer](h_2, data_2.edge_index, data_2.edge_attr)
            h_2 = F.elu(self.net_norms[layer](h_2, data_2.batch))
            
            h_global_1 = self.pool(h_1, data_1.batch)
            h_global_1 = self.feat_lin(h_global_1)
            out_global_1 = self.out_lin(h_global_1)
            h_sub_1 = self.pool(h_1, data_1.motif_batch)
            h_sub_1 = self.feat_lin(h_sub_1)
            
            h_global_2 = self.pool(h_2, data_2.batch)
            h_global_2 = self.feat_lin(h_global_2)
            out_global_2 = self.out_lin(h_global_2)
            h_sub_2 = self.pool(h_2, data_2.motif_batch)
            h_sub_2 = self.feat_lin(h_sub_2)
            
            sum_1 = 0
            sum_2 = 0
            # 这个片段会放在for loop外面
            new_h_sub_1 = h_sub_1.clone()
            new_h_sub_2 = h_sub_2.clone()

            for idx_1, idx_2 in zip(data_1.motif_num, data_2.motif_num):
                output_1, attention_weights_1 = dot_product_attention(h_sub_1[sum_1:sum_1 + idx_1, :], h_sub_2[sum_2:sum_2 + idx_2, :], h_sub_2[sum_2:sum_2 + idx_2, :])
                output_2, attention_weights_2 = dot_product_attention(h_sub_2[sum_2:sum_2 + idx_2, :], h_sub_1[sum_1:sum_1 + idx_1, :], h_sub_1[sum_1:sum_1 + idx_1, :])
                new_h_sub_1[sum_1:sum_1 + idx_1, :] = output_1
                new_h_sub_2[sum_2:sum_2 + idx_2, :] = output_2
                sum_1 += idx_1
                sum_2 += idx_2

            h_sub_1 = new_h_sub_1
            h_sub_2 = new_h_sub_2
            
            h_2 = add_graph_features_to_nodes(h_2, data_2.motif_batch, h_sub_2)
            h_1 = add_graph_features_to_nodes(h_1, data_1.motif_batch, h_sub_1)
        sum_1 = 0
        sum_2 = 0
        h_1_z = h_1
        h_2_z = h_2
        t1,t2 = new empty tensor
        for idx_1, idx_2 in zip(data_1.motif_num, data_2.motif_num):
            zitu_1 = h_sub_1[sum_1:sum_1 + idx_1, :] # zitu,dim
            zitu_1_gib = gib(zitu_1) # #zitu,dim
            # 经过一个类似sum,mean的操作使得每个图的维度变为  (1,dim)
            t1.append(zitu_1_gib)
            zitu_2 = h_sub_2[sum_2:sum_2 + idx_2, :]
            zitu_2_gib = gib(zitu_2)  
            t2.append(zitu_2_gib)
            # 经过一个类似sum,mean的操作使得每个图的维度变为  (1,dim)
        t1,t2  # bs,dim
        # batch_size dim
        h_global_1 = self.feat_lin(t1)
        h_global_2 = self.feat_lin(t2)

        return h_global_1, h_global_2

from torch_geometric.nn import Set2Set

class GraphInformationBottleneckModule(nn.Module):
    def __init__(self,
                device,
                node_input_dim=64,
                node_hidden_dim=64,
                num_step_set2set=2):
        super(GraphInformationBottleneckModule, self).__init__()
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.device = device
        
        self.compressor = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
        )

        self.predictor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.num_step_set2set = num_step_set2set
        self.set2set = Set2Set(self.node_hidden_dim, self.num_step_set2set)
        
    def compress(self, features):
        p = self.compressor(features)
        temperature = 1.0
        bias = 0.0001  # 防止数值问题
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, features):
        # 计算 lambda_pos 和 lambda_neg
        lambda_pos, p = self.compress(features)
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos

        # 获取 preserve_rate
        preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

        # 克隆并分离 features
        static_feature = features.clone().detach()

        # 计算均值和标准差
        node_feature_mean = static_feature.mean(dim=0, keepdim=True)
        node_feature_std = static_feature.std(dim=0, keepdim=True)

        # 生成噪声特征
        noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std
        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

        # 使用 set2set 方法处理噪声特征
        noisy_subgraphs = self.set2set(noisy_node_feature)

        # 计算 KL 损失
        epsilon = 1e-7
        KL_tensor = 0.5 * (((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim=1)) + \
                    ((((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2).mean(dim=1))
        KL_Loss = torch.mean(KL_tensor)

        return noisy_subgraphs, KL_Loss, preserve_rate, lambda_pos

# class GraphInformationBottleneckModule(nn.Module):
#     def __init__(self,
#                 device,
#                 node_input_dim=64,
#                 node_hidden_dim=64,
#                 num_step_set2set = 2):
#         super(GraphInformationBottleneckModule, self).__init__()
#         self.node_input_dim = node_input_dim
#         self.node_hidden_dim = node_hidden_dim
#         self.device = device
#         self.compressor = nn.Sequential(
#             nn.Linear(self.node_input_dim, self.node_hidden_dim),
#             nn.BatchNorm1d(self.node_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.node_hidden_dim, 1)
#             )

#         self.predictor = nn.Sequential(
#             nn.Linear(2*self.node_hidden_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64)
#         )
#         self.num_step_set2set = num_step_set2set
#         self.set2set = Set2Set(self.node_hidden_dim, self.num_step_set2set)
        
#     def compress(self, features):
#         p = self.compressor(features)
#         temperature = 1.0
#         bias = 0.0 + 0.0001  # If bias is 0, we run into problems
#         eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
#         gate_inputs = torch.log(eps) - torch.log(1 - eps)
#         gate_inputs = gate_inputs.to(self.device)
#         gate_inputs = (gate_inputs + p) / temperature
#         gate_inputs = torch.sigmoid(gate_inputs).squeeze()

#         return gate_inputs, p
    
#     def forward(self, features, bg):
#         # 计算 lambda_pos 和 lambda_neg
#         lambda_pos, p = self.compress(features)
#         lambda_pos = lambda_pos.reshape(-1, 1)
#         lambda_neg = 1 - lambda_pos

#         # 获取 preserve_rate
#         preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

#         # 克隆并分离 features
#         static_feature = features.clone().detach()
        
#         # 获取批次索引
#         batch_num_nodes = bg.batch_num_nodes()
#         batch_index = torch.cat([torch.full((num,), i, dtype=torch.long) for i, num in enumerate(batch_num_nodes)]).to(features.device)
        
#         # 调试输出以确保索引和特征的长度匹配
#         # print(f"features shape: {features.shape}")
#         # print(f"static_feature shape: {static_feature.shape}")
#         # print(f"batch_index shape: {batch_index.shape}")
        
#         # 计算均值和标准差
#         node_feature_mean = scatter_mean(static_feature, batch_index, dim=0)[batch_index]
#         node_feature_std = scatter_std(static_feature, batch_index, dim=0)[batch_index]

#         # 生成噪声特征
#         noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
#         noisy_node_feature_std = lambda_neg * node_feature_std
#         noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

#         # 使用 set2set 方法处理噪声特征
#         noisy_subgraphs = self.set2set(noisy_node_feature, batch_index)

#         # 计算 KL 损失
#         epsilon = 1e-7
#         KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim=1), batch_index).reshape(-1, 1) + \
#                     scatter_add((((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2), batch_index, dim=0)
#         KL_Loss = torch.mean(KL_tensor)
#         # print("KL loss:",KL_Loss)
#         # Prediction Y
#         final_features = noisy_subgraphs
#         return final_features, KL_Loss, preserve_rate, lambda_pos


if __name__ == "__main__":
    import sys
    sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
    from dataset.dataset import MoleculeDatasetWrapper
    model = GINet().to(torch.device("cuda:0"))
    print(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])

    dataset=MoleculeDatasetWrapper(2,1,0.2,"/home/dwj/WWW/DDIsubgraph/pretrain/pubchem-10m-clean-2w.txt")
    tran_loader, valid_loader=dataset.get_data_loaders()
    for step, batch in enumerate(tran_loader):
        batch=batch.to(torch.device("cuda:0"))
        print(batch)
        print(batch.motif_num)
        # print(batch.motif_batch)
        h_global, out_global, h_sub, out_sub=model(batch)
        print(h_global.shape)
        print(out_global.shape)
        # print(h_sub[0,batch.motif_num[0]:,:])
        print(h_sub.shape)
        print(out_sub.shape)
        # if(step!=0 and step%5==0):
        break