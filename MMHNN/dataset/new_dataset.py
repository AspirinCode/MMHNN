import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import itertools
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT

import sys
sys.path.append("/home/dwj/WWW/DDIsubgraph/pretrain")
from ps.mol_bpe import Tokenizer


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def read_smiles(data_path):
    smiles_data = []
    titanic_data = pd.read_csv(data_path,header=None)
    smiles_data=np.array(titanic_data)
    #这里我已经成功把他变成2元组了
    return smiles_data

def getSubGraph(smi):
    # 用主子图方法提取分子的子图
    # 输入smiles
    tokenizer = Tokenizer('/home/dwj/WWW/DDIsubgraph/pretrain/vocabulary_pubchem_100w_300.txt')
    groups = tokenizer(smi)  # 获取子图的id
    return groups
#函数引用：
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature

def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    return undirected_edge_list.T, features


# 构建分子数据集
class MoleculeDataset(Dataset):
    def __init__(self, smiles_data,max_len=16):
        super(Dataset, self).__init__()
        self.smiles_data = smiles_data  # smiles序列
        self.max_len=max_len  # 子图最大数量

    def __getitem__(self, index):
        print(self.smiles_data[0])
        mol = Chem.MolFromSmiles(self.smiles_data[index][0])
        # mol = Chem.AddHs(mol)
        
        N = mol.GetNumAtoms()  # 原子数量
        M = mol.GetNumBonds()  # 键数量

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
        #将在这里把先把编码情况给改了
        features = [(atom.GetIdx(), atom_features(atom)) for atom in mol.GetAtoms()]
        features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
        _, features = zip(*features)
        features = torch.stack(features)
        x=features
        '''
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        ATOM_MAX_NUM = mol.GetNumAtoms()
        AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(mol.GetAtoms())})
        AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(mol.GetAtoms())})
        max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(mol.GetAtoms()))
        max_valence = max(max_valence, 9)
        MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(mol.GetAtoms())]))
        MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
        MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(mol.GetAtoms())]))
        MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0
        
        get_mol_edge_list_and_feat_mtx(mol)
        '''
        
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

        group_ids, res = getSubGraph(self.smiles_data[index])
        res = sorted(res, key=lambda t: len(t[1]), reverse=True)
        group_ids = sorted(group_ids, key=len, reverse=True)  # 将子图id按照子图大小 从大到小排列
        if (len(group_ids) > self.max_len):  # 将大于max_len的部分截去
            group_ids = group_ids[:self.max_len]
            res = res[:self.max_len]
        sub_smiles = [x[0] for x in res]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #将每一组数据凝练成一组列表
        data1=[data, group_ids, N,self.max_len]
        
        
        mol = Chem.MolFromSmiles(self.smiles_data[index][1])
        # mol = Chem.AddHs(mol)
        
        N = mol.GetNumAtoms()  # 原子数量
        M = mol.GetNumBonds()  # 键数量

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
        #将在这里把先把编码情况给改了
        features = [(atom.GetIdx(), atom_features(atom)) for atom in mol.GetAtoms()]
        features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
        _, features = zip(*features)
        features = torch.stack(features)
        x=features
        '''
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        ATOM_MAX_NUM = mol.GetNumAtoms()
        AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(mol.GetAtoms())})
        AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(mol.GetAtoms())})
        max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(mol.GetAtoms()))
        max_valence = max(max_valence, 9)
        MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(mol.GetAtoms())]))
        MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
        MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(mol.GetAtoms())]))
        MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0
        
        get_mol_edge_list_and_feat_mtx(mol)
        '''
        
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

        group_ids, res = getSubGraph(self.smiles_data[index])
        res = sorted(res, key=lambda t: len(t[1]), reverse=True)
        group_ids = sorted(group_ids, key=len, reverse=True)  # 将子图id按照子图大小 从大到小排列
        if (len(group_ids) > self.max_len):  # 将大于max_len的部分截去
            group_ids = group_ids[:self.max_len]
            res = res[:self.max_len]
        sub_smiles = [x[0] for x in res]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # 返回分子数据，子图id，原子数量，最大子图数量
        data2=[data, group_ids, N,self.max_len]
        fin_data=[data1,data2,self.smiles_data[index][2],self.smiles_data[index][3]]

        return fin_data

    def __len__(self):
        return len(self.smiles_data)


def collate_fn(batch):
    fin_data=batch
    data1,data2,label,r=fin_data[0],fin_data[1],fin_data[2],fin_data[3]
    
    datas, graphs_ids,atom_nums, max_len = data1[0],data1[1],data1[2],data1[3]

    datas = Batch.from_data_list(datas)  # 构建分子图batch

    datas.motif_batch = torch.zeros(datas.x.size(0), dtype=torch.long)  # 增加一个子图batch，代表每个子图属于哪个motif

    batch_size=len(graphs_ids)
    datas.motif_num = torch.zeros(batch_size, dtype=torch.long)  # 增加一个字段，保存每个分子所含子图数目
    datas.max_len=torch.tensor(max_len[0])  # 增加一个字段，保存最大子图数量

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

    datas.mask = torch.ones((batch_size, max_len[0]), dtype=torch.bool)  # 增加一个mask字段
    for batch_id in range(batch_size):
        id = torch.randint(0, datas.motif_num[batch_id], size=(1,))  # 在子图总数内随机生成一个被mask的子图
        datas.mask[batch_id][id] = 0
        
        
    sp_1=datas 
    datas, graphs_ids,atom_nums, max_len = data2[0],data2[1],data2[2],data2[3]

    datas = Batch.from_data_list(datas)  # 构建分子图batch

    datas.motif_batch = torch.zeros(datas.x.size(0), dtype=torch.long)  # 增加一个子图batch，代表每个子图属于哪个motif

    batch_size=len(graphs_ids)
    datas.motif_num = torch.zeros(batch_size, dtype=torch.long)  # 增加一个字段，保存每个分子所含子图数目
    datas.max_len=torch.tensor(max_len[0])  # 增加一个字段，保存最大子图数量

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

    datas.mask = torch.ones((batch_size, max_len[0]), dtype=torch.bool)  # 增加一个mask字段
    for batch_id in range(batch_size):
        id = torch.randint(0, datas.motif_num[batch_id], size=(1,))  # 在子图总数内随机生成一个被mask的子图
        datas.mask[batch_id][id] = 0
    data=[sp_1,datas,lable,r]
    return data


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, max_len=16):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.max_len=max_len
        self.length=0

    def get_data_loaders(self):
        smiles_data = read_smiles(self.data_path)

        num_train = len(smiles_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        self.length = num_train

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_smiles = [smiles_data[i] for i in train_idx]
        valid_smiles = [smiles_data[i] for i in valid_idx]
        del smiles_data
        print(len(train_smiles), len(valid_smiles))

        train_dataset = MoleculeDataset(train_smiles, max_len=self.max_len)
        #valid_dataset = MoleculeDataset(valid_smiles, max_len=self.max_len)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True,collate_fn=collate_fn, shuffle=True
        )
        return train_loader
class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    dataset=MoleculeDatasetWrapper(batch_size=32, num_workers=8, valid_size=0.2, data_path='/home/dwj/WWW/DDIsubgraph/pretrain/pubchem-10m-clean-100w.txt', max_len=16)
    tran_loader, valid_loader=dataset.get_data_loaders()
    for step, mol_batch in enumerate(tran_loader):
        print(mol_batch)
        # print(mol_batch.mask)
        # print(mol_batch.motif_to_mol)
        # print(group_id_batch[0])
        # print(group_id_batch[0][0])
        break