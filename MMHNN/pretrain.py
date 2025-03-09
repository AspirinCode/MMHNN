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

import sys
sys.path.append("/home/ljh/remote/my-project/pretrain")
from dataset.dataset import MoleculeDatasetWrapper
from models.model import Model001
from loss_utils.nt_xent import NTXentLoss
from loss_utils.weighted_nt_xent import Weighted_NTXentLoss
from loss_utils.motif_loss import Motif_Loss


def train(args, model, train_loader, valid_loader, optimizer, device, gama):
    model.train()
    xent_list = []
    motif_list = []
    loss_list = []

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch=batch.to(device)
        out_global, motif_embeddings, weight, weight_mask, out_sub, motif_num = model(batch)
        nt_xent_criterion = NTXentLoss(device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.5, lambda_2=0.5)
        nt_xent_criterion_weighted = Weighted_NTXentLoss(device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.5, lambda_2=0.5)
        motif_criterion = Motif_Loss(device, use_cosine_similarity=True)

        out_global = F.normalize(out_global, dim=1)
        motif_embeddings = F.normalize(motif_embeddings, dim=1)
        if args.weight==1:
            xent_loss = nt_xent_criterion_weighted(out_global, motif_embeddings, weight_mask)
        else:
            xent_loss=nt_xent_criterion(out_global, motif_embeddings)
        motif_loss = motif_criterion(out_sub, motif_num)
        loss = xent_loss+gama*motif_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        xent_list.append(xent_loss.item())
        motif_list.append(motif_loss.item())
        loss_list.append(loss.item())

    train_xent_loss = sum(xent_list)/len(xent_list)
    train_motif_loss = sum(motif_list)/len(motif_list)
    train_loss = sum(loss_list)/len(loss_list)
    print("train_loss:",train_loss,"train_xent_loss:",train_xent_loss,"train_motif_loss:",train_motif_loss)

    valid_loss = validate(args,model,valid_loader,device,gama)
    return train_loss, valid_loss


def validate(args,model, valid_loader,device,gama):
    with torch.no_grad():
        model.eval()
        xent_list = []
        motif_list = []
        loss_list = []
        for step, batch in enumerate(tqdm(valid_loader, desc="Iteration")):
            batch = batch.to(device)
            out_global, motif_embeddings, weight, weight_mask, out_sub, motif_num = model(batch)
            nt_xent_criterion = NTXentLoss(device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.5,
                                           lambda_2=0.5)
            nt_xent_criterion_weighted = Weighted_NTXentLoss(device, temperature=0.1, use_cosine_similarity=True,
                                                             lambda_1=0.5, lambda_2=0.5)
            motif_criterion = Motif_Loss(device, use_cosine_similarity=True)

            out_global = F.normalize(out_global, dim=1)
            motif_embeddings = F.normalize(motif_embeddings, dim=1)
            if args.weight == 1:
                xent_loss = nt_xent_criterion_weighted(out_global, motif_embeddings, weight_mask)
            else:
                xent_loss = nt_xent_criterion(out_global, motif_embeddings)
            motif_loss = motif_criterion(out_sub, motif_num)
            loss = xent_loss + gama * motif_loss

            xent_list.append(xent_loss.item())
            motif_list.append(motif_loss.item())
            loss_list.append(loss.item())

        valid_xent_loss = sum(xent_list) / len(xent_list)
        valid_motif_loss = sum(motif_list) / len(motif_list)
        valid_loss = sum(loss_list)/len(loss_list)
        print("valid_loss:", valid_loss, "valid_xent_loss:", valid_xent_loss, "valid_motif_loss:", valid_motif_loss)
    model.train()
    return valid_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help=' inputbatch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default='/home/ljh/remote/data/pubchem-10m-clean-100w.txt',
                        help='root directory of dataset.')

    parser.add_argument('--valid_size', type=float, default=0.05,
                        help='valid_size (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help=' the number of workers to load data (default: 8)')


    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--feat_dim', type=int, default=256,
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

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--decay', type=float, default=0.00001,
                        help='weight decay (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.001)')


    parser.add_argument('--output_model_file', type=str, default='/home/ljh/remote/my-project/pretrain/save_model/pretrain/motif_loss+weight',
                        help='filename to output the pre-trained model')
    parser.add_argument('--gama', type=float, default=0.2, help='weight of motif_loss')
    args = parser.parse_args()

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
                     dropout_encoder=args.dropout_encoder).to(device)         # model
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 9, eta_min=0, last_epoch=-1)

    min_loss = 1000000
    train_loss_list=[]
    valid_loss_list=[]
    result_dir=args.output_model_file+"/"+"gama-"+str(args.gama)+"_batch_size-"+str(args.batch_size)+"_dataset-"+\
               str(dataset.length)+"_dropout_encoder-"+str(args.dropout_encoder)+"_N-"+str(args.N)+"_num_layer-"+\
               str(args.num_layer)+"_weight-"+str(args.weight)    # 存的输出模型的路径
    os.makedirs(result_dir, exist_ok=True)

    a=validate(args, model, train_loader, device, args.gama)
    b=validate(args, model, valid_loader, device, args.gama)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        epoch_start=time.time()
        train_loss,valid_loss = train(args, model, train_loader, valid_loader, optimizer, device, args.gama)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        epoch_end=time.time()

        print("epoch:", epoch, "time:", epoch_end-epoch_start, "s")

        if not args.output_model_file == "":
            torch.save(model.state_dict(), result_dir + "/pretrain_model_epoch_" + str(epoch) + ".pth")   # 保存模型
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), result_dir + "/pretrain_motif_model_min_loss_" + ".pth")

        if epoch >= 10:
            print("warmup")
            scheduler.step()

    plt.plot(train_loss_list)
    plt.savefig(result_dir+'/train_loss_list.png')   #loss下降图

    plt.clf()

    plt.plot(valid_loss_list)
    plt.savefig(result_dir + '/valid_loss_list.png')








if __name__ == "__main__":
    main()