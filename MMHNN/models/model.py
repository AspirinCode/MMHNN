import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import sys
sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
from models.gnn import GINet
from models.transformer import make_model
import warnings

warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
class Model001(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, dropout_gin=0, pool='mean', device=torch.device("cuda:0"),
                 N=6, d_model=256, d_ff=1024, h=8, dropout_encoder=0.1):
        super(Model001, self).__init__()
        self.device=device
        self.gin=GINet(num_layer=num_layer,emb_dim=emb_dim,feat_dim=feat_dim,dropout=dropout_gin,pool=pool,device=device)
        self.encoder=make_model(N=N,d_model=d_model,d_ff=d_ff,h=h,dropout=dropout_encoder)
        self.fc1 = nn.Linear(emb_dim*2, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self,batch,device):
        graph_1,  graph_2, label = batch
        graph_1 = graph_1.to(device)
        graph_2 = graph_2.to(device)
        
        h_global_1,h_global_2=self.gin(graph_1,graph_2)
        h_global = torch.cat((h_global_1,h_global_2), dim=1)
        h_global = F.relu(self.fc1(h_global))
        h_global = F.relu(self.fc2(h_global))
        x = self.fc3(h_global)
        return x
    def pretrain_result(self, data):
        h_global, out_global, h_sub, out_sub=self.gin(data)
        return out_global,out_sub



if __name__ == "__main__":
    import sys
    sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
    from dataset.dataset import MoleculeDatasetWrapper
    model = Model001().to(torch.device("cuda:0"))
    # print(model)
    dataset=MoleculeDatasetWrapper(2,1,0.2,"/home/dwj/WWW/DDIsubgraph/pubchem-10m-clean-2w.txt")
    tran_loader, valid_loader=dataset.get_data_loaders()
    for step, batch in enumerate(tran_loader):
        batch=batch.to(torch.device("cuda:0"))
        # print(batch)
        # print(batch.motif_num)

        out_global, motif_embeddings, weight, weight_mask, out_sub, motif_num=model(batch)
        # print(weight_mask)
        # print(1-weight_mask)
        print("batch_mask:",batch.mask)
        print("out_global:",out_global.shape)
        print("motif_embeddings:",motif_embeddings.shape)
        print("weight:",weight)
        print("weight_mask:",weight_mask)
        print("out_sub:",out_sub.shape)
        print("motif_num:",motif_num)

        break