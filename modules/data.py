# From PyGAS, PyTorch Geometric Auto-Scale: https://github.com/rusty1s/pyg_autoscale/tree/master
from typing import Tuple

import scipy.io
import torch
import numpy as np
from torch.utils.data import random_split
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import (PPI, Amazon, Coauthor, Flickr,
                                      GNNBenchmarkDataset, Planetoid, Reddit2,
                                      WikiCS, Yelp, DBLP)
from .utils import gen_masks, index2mask
from .linkx.dataset import load_snap_patents_mat


def get_blogcat(root: str, name: str, split_id: int=0) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/blogcatalog_0.6/split_{split_id}.pt')
    graph = scipy.io.loadmat(f'{root}/blogcatalog_0.6/blogcatalog.mat')
    edges = graph['network'].nonzero()
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    train_mask = torch.zeros(10312, dtype=torch.bool)
    test_mask = torch.zeros(10312, dtype=torch.bool)
    val_mask = torch.zeros(10312, dtype=torch.bool)
    train_mask[dataset['train_mask']] = True
    test_mask[dataset['test_mask']] = True
    val_mask[dataset['val_mask']] = True

    data = Data(y=torch.tensor(graph['group'].todense()), edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
                num_classes=39, num_nodes=10312)

    return data, data.num_features, data.num_classes


def get_dblp(root: str, name: str, seed: int = None) -> Tuple[Data, int, int]:
    features = []
    with open(f'{root}/features.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            feature = value.split(',')
            number = [float(i) for i in feature]
            features.append(number)
    edges = []
    with open(f'{root}/dblp.edgelist', 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            source_node = int(parts[0])
            target_node = int(parts[1])
            edges.append((source_node, target_node))
            edges.append((source_node, target_node))

    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    labels = []
    with open(f'{root}/labels.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            label = value.split(',')
            number = [1. if float(x)==1 else 0. for x in label]
            labels.append(number)
    torch.manual_seed(seed)
    num_nodes = 28702
    train_ratio = 0.8
    test_ratio = 0.1
    num_train = int(train_ratio * num_nodes)
    num_test = int(test_ratio * num_nodes)
    num_val = num_nodes - num_train - num_test

    train_idx, test_idx, val_idx = random_split(list(range(num_nodes)), [num_train, num_test, num_val])
    train_mask = torch.zeros(28702, dtype=torch.bool)
    test_mask = torch.zeros(28702, dtype=torch.bool)
    val_mask = torch.zeros(28702, dtype=torch.bool)
    train_mask[train_idx.indices] = True
    test_mask[test_idx.indices] = True
    val_mask[val_idx.indices] = True

    data = Data(edge_index=edge_index, x=torch.tensor(features),y=torch.tensor(labels), train_mask=train_mask,
                test_mask=test_mask, val_mask=val_mask, num_features=300)
    data.node_stores[0].num_classes = 4
    return data, data.num_features, data.num_classes


def get_synth(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/Hyperspheres_10_10_0_0.6/split_0.pt')
    edge_index = torch.load(f'{root}/Hyperspheres_10_10_0/edge_index.pt')
    labels = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/labels.csv', skip_header=1,
                       dtype=np.dtype(float), delimiter=','))
    features = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/features.csv', skip_header=1,
                       dtype=np.dtype('float32'), delimiter=','))
    data = Data(x=features, y=labels, edge_index=edge_index, train_mask=dataset['train_mask'],
                test_mask=dataset['test_mask'], val_mask=dataset['val_mask'],
                num_classes=20)
    return data, data.num_features, data.num_classes


def get_synth_high(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/Hyperspheres_10_10_0_0.6/split_0.pt')
    edge_index = torch.load(f'{root}/Hyperspheres_10_10_0/edge_index_high.pt')
    labels = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/labels.csv', skip_header=1,
                       dtype=np.dtype(float), delimiter=','))
    features = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/features.csv', skip_header=1,
                       dtype=np.dtype('float32'), delimiter=','))
    data = Data(x=features, y=labels, edge_index=edge_index, train_mask=dataset['train_mask'],
                test_mask=dataset['test_mask'], val_mask=dataset['val_mask'],
                num_classes=20)
    return data, data.num_features, data.num_classes


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(f'{root}/Planetoid', name, split='full') #, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(f'{root}/WIKICS', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor(remove_edge_index = False))
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor(remove_edge_index = False))
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB')#,
                                     #pre_transform=T.ToSparseTensor())
    data = dataset[0]
    #data.adj_t = data.adj_t.to_symmetric()
    # Instead of the commented line above use the following line to convert the edge_index to undirected
    data.edge_index = pygutils.to_undirected(data.edge_index)
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB')#,
                                     #pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP')#, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr')#, pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2')#, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_ppi(root: str, split: str = 'train') -> Tuple[Data, int, int]:
    dataset = PPI(f'{root}/PPI', split=split)#, pre_transform=T.ToSparseTensor())
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
    return data, dataset.num_features, dataset.num_classes


def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=T.ToSparseTensor())
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes


def get_proteins(root: str):
    dataset = PygNodePropPredDataset('ogbn-proteins', root)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    # This is a multi-label binary classification dataset, so we need
    # float targets for BCEWithLogitsLoss
    data.y = data.y.float()

    return data, dataset.num_features, data.y.shape[1]


def get_linkx_dataset(root: str, name: str, seed: int = None):
    if name.lower() == 'snap-patents':
        dataset = load_snap_patents_mat(root)
        split_idx = dataset.get_idx_split(seed=seed)
        num_nodes = dataset.graph['num_nodes']
        train_mask = index2mask(split_idx['train'], num_nodes)
        valid_mask = index2mask(split_idx['valid'], num_nodes)
        test_mask = index2mask(split_idx['test'], num_nodes)

        edge_index = dataset.graph['edge_index']
        edge_index = pygutils.to_undirected(edge_index, num_nodes=num_nodes)

        data = Data(x=dataset.graph['node_feat'],
                    edge_index=edge_index,
                    y=dataset.label,
                    train_mask=train_mask,
                    val_mask=valid_mask,
                    test_mask=test_mask)
        num_classes = len(data.y.unique())
    else:
        raise ValueError(f'Unknown dataset name: {name}')

    return data, data.num_features, num_classes

def get_data(root: str,
             name: str,
             seed: int = None,
             split_id: int = 0,
             ) -> Tuple[Data, int, int]:
    if name.lower() in ['blogcat']:
        return get_blogcat(root, name, split_id)
    elif name.lower() == 'dblp':
        return get_dblp(root, name, seed)
    elif name.lower() == 'synth':
        return get_synth(root, name)
    elif name.lower() == 'synth_high':
        return get_synth_high(root, name)
    elif name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name)
    elif name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:])
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:])
    elif name.lower() == 'wikics':
        return get_wikics(root)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() == 'ppi':
        return get_ppi(root)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() == 'yelp':
        return get_yelp(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root)
    elif name.lower() == 'ogbn-proteins':
        return get_proteins(root)
    elif name.lower() == 'snap-patents':
        return get_linkx_dataset(root, 'snap-patents', seed)
    else:
        raise NotImplementedError
