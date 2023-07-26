from typing import Union

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

class GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int],
                 normalize):
        super(GCN, self).__init__()

        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1],
                                      normalize=normalize))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1], normalize=normalize))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                edge_weight: Union[torch.Tensor, list[torch.Tensor]],
                add_self_loops: bool
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            weight = torch.tensor(edge_weight[-i]) if layerwise_adjacency else edge_weight
            edge_indices, edge_weights = gcn_norm(edges, weight, add_self_loops=add_self_loops)
            # weight = torch.cat((weight, torch.ones(edge_indices.size(1) - edges.size(1), device=weight.device)))
            # weight = weight*edge_weights

            x = torch.relu(layer(x, edge_indices, edge_weights))

        edges = edge_index[0] if layerwise_adjacency else edge_index
        weight = torch.tensor(edge_weight[0]) if layerwise_adjacency else edge_weight
        edge_indices, edge_weights = gcn_norm(edges, weight, add_self_loops=add_self_loops)
        # weight = torch.cat((weight, torch.ones(edge_indices.size(1) - edges.size(1), device=weight.device)))
        # weight = weight * edge_weights
        logits = self.gcn_layers[-1](x, edge_indices, edge_weights)

        return logits


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=None, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = row if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[row]
