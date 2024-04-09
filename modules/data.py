# From PyGAS, PyTorch Geometric Auto-Scale: https://github.com/rusty1s/pyg_autoscale/tree/master
from typing import Tuple

import scipy.io
import torch
import numpy as np
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import (PPI, Amazon, Coauthor, Flickr,
                                      GNNBenchmarkDataset, Planetoid, Reddit2,
                                      WikiCS, Yelp)

from .utils import gen_masks, index2mask

def get_blogcat(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/blogcatalog_0.6/split_0.pt')
    graph = scipy.io.loadmat(f'{root}/blogcatalog_0.6/blogcatalog.mat')
    edges = graph['network'].nonzero()
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    data = Data(y=torch.tensor(graph['group'].todense()), edge_index=edge_index,
                train_mask=dataset['train_mask'], test_mask=dataset['test_mask'], val_mask=dataset['val_mask'],
                num_classes=39)
    data.node_stores[0].x = torch.empty(data.num_nodes, 32)

    return data, data.num_features, data.num_classes


def get_dblp(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = DBLP(f'{root}/dblp')
    return dataset[0], dataset.num_features, dataset.num_classes


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
    dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor())
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor())
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


def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['blogcat']:
        return get_blogcat(root, name)
    elif name.lower() == 'dblp':
        return get_dblp(root, name)
    elif name.lower() == 'synth':
        return get_synth(root, name)
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
    else:
        raise NotImplementedError
