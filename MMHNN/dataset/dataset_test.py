import os
import csv
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import sys
sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
from ps.mol_bpe import Tokenizer
from tqdm import tqdm



ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    print('train: {}, valid: {}, test: {}'.format(
        len(train_inds), len(valid_inds), len(test_inds)))
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row[3]
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print('Number of data:', len(smiles_data))
    return smiles_data, labels


def getSubGraph(smi):
    # 用主子图方法提取分子的子图
    # 输入smiles
    tokenizer = Tokenizer('/home/dwj/WWW/DDIsubgraph/pretrain/vocabulary_pubchem_100w_300.txt')
    groups, res = tokenizer(smi)  # 获取子图的id
    return groups,res



class MolTestDataset(Dataset):
    def __init__(self, smiles_data, labels, target='p_np', task='classification', max_len=16):
        super(Dataset, self).__init__()
        self.smiles_data = smiles_data
        self.labels = labels
        self.task = task
        self.max_len = max_len  # 子图最大数量

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1, -1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)


        group_ids, res=getSubGraph(self.smiles_data[index])
        res=sorted(res, key=lambda t: len(t[1]), reverse=True)
        group_ids=sorted(group_ids, key=len, reverse=True)  # 将子图id按照子图大小 从大到小排列
        if(len(group_ids)>self.max_len):  # 将大于max_len的部分截去
            group_ids=group_ids[:self.max_len]
            res=res[:self.max_len]
        sub_smiles=[x[0] for x in res]


        with open("/home/dwj/WWW/DDIsubgraph/pretrain/vocabulary_pubchem_100w_300.txt", 'r') as fin:
            smis = list(map(lambda x: x.strip(), fin.readlines()))
        smis = smis[1:]
        smis = [smi.split("\t", 1)[0] for smi in smis]
        try:
            sub_smi_ids=[smis.index(smi) for smi in sub_smiles]
            sub_smi_ids=sub_smi_ids+[-1]*(self.max_len-len(sub_smi_ids))
        except:
            sub_smi_ids=[-1]*self.max_len

        # print(group_ids)
        # print(sub_smiles)
        # print(sub_smi_ids)
        return data, group_ids, N,self.max_len, sub_smi_ids

    def __len__(self):
        return len(self.smiles_data)


def collate_fn(batch):
    datas, graphs_ids,atom_nums, max_len, sub_smi_ids = zip(*batch)
    # print(len(sub_smi_ids))

    datas = Batch.from_data_list(datas)  # 构建分子图batch
    # print(datas)

    datas.motif_batch = torch.zeros(datas.x.size(0), dtype=torch.long)  # 增加一个子图batch，代表每个子图属于哪个motif
    # print(datas)

    batch_size=len(graphs_ids)
    datas.motif_num = torch.zeros(batch_size, dtype=torch.long)  # 增加一个字段，保存每个分子所含子图数目
    datas.max_len=torch.tensor(max_len[0])  # 增加一个字段，保存最大子图数量
    # print(datas)
    datas.subgraph_to_num=torch.tensor(sub_smi_ids)

    curr_indicator = 0
    curr_num = 0
    k = 0
    for N, indices in zip(atom_nums, graphs_ids):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            datas.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        datas.motif_num[k] = len(indices)
        curr_num += N
        k += 1
    # print(datas)
    return datas



class MolTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task, max_len):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.max_len=max_len
        self.smiles_data, self.labels = read_smiles(data_path, target, task)


    def get_data_loaders(self):

        train_dataset = MolTestDataset(smiles_data=self.smiles_data, labels=self.labels, target=self.target, task=self.task, max_len=self.max_len)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn,shuffle=False)

        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn,shuffle=False)

        return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    dataset=MolTestDatasetWrapper(batch_size=32, num_workers=8, valid_size=0.1,test_size=0.1,data_path="/home/dwj/WWW/DDIsubgraph/pretrain/bbbp/BBBP.csv",target="p_np",task="classification", max_len=16)
    train_loader, valid_loader, test_loader=dataset.get_data_loaders()
    for i in range(100):
        print("epoch:",i)
        for step, mol_batch in enumerate(tqdm(valid_loader, desc="Iteration")):
            print(mol_batch)