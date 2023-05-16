import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torch import Tensor


class TargetNeighborhoodDataset(Dataset):
    def __init__(self,
                 target_nodes: Tensor,
                 edge_index: Tensor,
                 num_nodes: int,
                 num_hops: int):
        self.target_nodes = target_nodes
        num_edges = edge_index.shape[-1]
        values = np.ones(num_edges, dtype=bool)
        self.adjacency = sp.csr_matrix((values, edge_index),
                                       shape=(num_nodes, num_nodes))
        self.num_hops = num_hops

    def __getitem__(self, item):
        return self.target_nodes[item]

    def __len__(self):
        return self.target_nodes.shape[-1]

    def collate_with_neighborhoods(self, target_nodes):
        nodes = torch.stack(target_nodes)
        all_hops = []
        for hop in range(self.num_hops):
            neighborhood = self.adjacency[nodes].tocoo()
            edge_index = torch.stack([nodes[neighborhood.row],
                                      torch.tensor(neighborhood.col)],
                                     dim=0)
            all_hops.append(edge_index)

        return all_hops
