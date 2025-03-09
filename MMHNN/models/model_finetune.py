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

class Model_Finetune(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, dropout_gin=0, pool='mean', device=torch.device("cuda:0"),task='classification',
                 N=6, d_model=256, d_ff=1024, h=8, dropout_encoder=0.1):
        super(Model_Finetune, self).__init__()
        self.device=device
        self.task=task
        self.gin=GINet(num_layer=num_layer,emb_dim=emb_dim,feat_dim=feat_dim,dropout=dropout_gin,pool=pool,device=device)
        self.encoder=make_model(N=N,d_model=d_model,d_ff=d_ff,h=h,dropout=dropout_encoder)

        # projection head
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim//2, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),  # first layer
            # nn.Linear(feat_dim, feat_dim, bias=False),
            # nn.BatchNorm1d(feat_dim),
            # nn.ReLU(inplace=True),  # second layer
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim)
        )

        # fine-tune prediction layers
        if self.task == 'classification':
            self.output_layers = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.Softplus(),
                nn.Linear(feat_dim // 2, 2)
            )
        elif self.task == 'regression':
            self.output_layers = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.Softplus(),
                nn.Linear(feat_dim // 2, 1)
            )
        else:
            raise ValueError('Undefined task type!')


    def forward(self, data):
        h_global, out_global, h_sub, out_sub=self.gin(data)
        weight = self.encoder(h_sub,mask=None)

        out_sub_weighted = weight.unsqueeze(-1)*out_sub

        motif_embeddings = out_sub_weighted.sum(dim=1)

        h = self.proj_head(motif_embeddings)

        return self.output_layers(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('NOT LOADED:', name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)



if __name__ == "__main__":
    import sys
    sys.path.append("//home/dwj/WWW/DDIsubgraph/pretrain")
    from dataset.dataset_test import MolTestDatasetWrapper
    model = Model_Finetune().to(torch.device("cuda:0"))
    # print(model)
    state_dict = torch.load("//home/dwj/WWW/DDIsubgraph/pretrain/save_model/pretrain/motif_loss+weight/gama-1_batch_size-64_dataset-20000/pretrain_motif_model_min_loss_.pth", map_location=torch.device("cuda:0"))
    model.load_my_state_dict(state_dict)
    print(model)

    dataset = MolTestDatasetWrapper(batch_size=2, num_workers=8, valid_size=0.1, test_size=0.1,
                                    data_path="/home/ljh/remote/data/bbbp/BBBP.csv", target="p_np",
                                    task="classification", max_len=16)
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    for step, batch in enumerate(train_loader):
        batch=batch.to(torch.device("cuda:0"))
        print(batch)

        h=model(batch)
        print(h)

        break