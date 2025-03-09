#导入模型参数
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
import sys
sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
from dataset.dataset import MoleculeDatasetWrapper
from models.model_new import Model001
from loss_utils.nt_xent import NTXentLoss
from loss_utils.weighted_nt_xent import Weighted_NTXentLoss
from loss_utils.motif_loss import Motif_Loss
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=1,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=128,
                    help=' inputbatch size for training (default: 1024)')
parser.add_argument('--dataset', type=str, default='/home/dwj/WWW/DDIsubgraph/pretrain/data/Compsol.csv',
                    help='root directory of dataset.')

parser.add_argument('--valid_size', type=float, default=0.05,
                    help='valid_size (default: 0.2)')
parser.add_argument('--num_workers', type=int, default=0,
                    help=' the number of workers to load data (default: 8)')


parser.add_argument('--num_layer', type=int, default=4,
                    help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='embedding dimensions (default: 300)')
parser.add_argument('--feat_dim', type=int, default=300,
                    help='embedding dimensions (default: 256)')
parser.add_argument('--dropout_gin', type=float, default=0,
                    help='dropout ratio (default: 0.2)')
parser.add_argument('--graph_pooling', type=str, default="mean",
                    help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--N', type=int, default=2,
                    help='num layer of transformer encoder')
parser.add_argument('--d_model', type=int, default=256,
                    help='embedding dimensions (default: 256)')
parser.add_argument('--d_ff', type=int, default=1024,
                    help='embedding dimensions (default: 1024)')
parser.add_argument('--h', type=int, default=8,
                    help='heads of transformer encoder(default: 8)')
parser.add_argument('--dropout_encoder', type=float, default=0.1,
                    help='dropout ratio (default: 0.1)')
parser.add_argument('--weight', type=int, default=1,
                    help='weight or not')

parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 100)')

parser.add_argument('--decay', type=float, default=0.00001,
                    help='weight decay (default: 0)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.001)')


parser.add_argument('--output_model_file', type=str, default='/home/dwj/WWW/DDIsubgraph/pretrain/save_model/pretrain/motif_loss+weight',
                    help='filename to output the pre-trained model')
parser.add_argument('--gama', type=float, default=0.2, help='weight of motif_loss')
args = parser.parse_args([])

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
dataset = MoleculeDatasetWrapper(args.batch_size, args.num_workers, args.valid_size, args.dataset)   # dataset：一个txt文件路径，里面村的是smiles
train_loader, valid_loader = dataset.get_data_loaders()  # 导入数据集，编码，在dataset里
model = Model001(num_layer=args.num_layer, emb_dim=args.emb_dim, feat_dim=args.feat_dim,
                 dropout_gin=args.dropout_gin, pool=args.graph_pooling, device=device,
                 N=args.N, d_model=args.d_model, d_ff=args.d_ff, h=args.h,
                 dropout_encoder=args.dropout_encoder).to(device)   


from datetime import datetime
import random
import pandas as pd
import numpy as np
import torch
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader
from sklearn import metrics

import layers1
import models1
import custom_loss
import time
import torch.nn as nn
#'C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
#            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
#            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'


# Hyperparameters
n_atom_feats = 55
n_atom_hid = 256
rel_total = 86
lr = 1e-3
weight_decay = 5e-4
n_epochs = 300
neg_samples = 1
batch_size = 1024
data_size_ratio = 1
kge_dim = 384

def rmse_loss(pred, target):
    labels_tensor = torch.tensor([float(x) for x in target], dtype=torch.float32).unsqueeze(1).to(pred.device)
    return F.mse_loss(pred, labels_tensor )
def mae_loss(pred, target):
    # 将目标值转换为浮点数张量，并调整形状以匹配预测值
    labels_tensor = torch.tensor([float(x) for x in target], dtype=torch.float32).unsqueeze(1).to(pred.device)
    # 计算绝对误差
    absolute_errors = torch.abs(pred - labels_tensor)
    # 返回平均绝对误差
    return torch.mean(absolute_errors)

def train(model, train_data_loader, val_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0

        sum1=0
        for batch in train_data_loader:
            model.train()
            p_score = model(batch,device)
            loss= rmse_loss(p_score,batch[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum1+=len(batch[2]) 
            loss1 = mae_loss(p_score,batch[2])
            train_loss += loss1.item() * len(p_score)
        train_loss /= sum1

        with torch.no_grad():
            sum2=0
            for batch in val_data_loader:
                model.eval()
                p_score = model(batch,device)
                loss= rmse_loss(p_score,batch[2]) 
                #print(len(batch[2])  ) 
                 
                sum2+=len(batch[2])
                loss1 = mae_loss(p_score,batch[2])       
                val_loss += loss1.item() * len(p_score)
            val_loss /= sum2
               
        if scheduler:
            print('scheduling')
            scheduler.step()
        print(f"train_loss:  {train_loss} and val_loss:  {val_loss} /n")
        
        
        

loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

train(model, train_loader, valid_loader, loss, optimizer, n_epochs, device, scheduler)

