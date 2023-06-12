from typing import Union

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add


class GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int]):
        super(GCN, self).__init__()

        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1],
                                      normalize=True))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1], normalize=True))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                edge_weight: Union[torch.Tensor, list[torch.Tensor]]
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            weight = torch.tensor(edge_weight[-i]) if layerwise_adjacency else edge_weight
            # normalize the edge weights based on node degrees (and exclude the added weights in the normalization)
            edge_weight = torch.ones((edges.size(1),), dtype=x.dtype,
                                     device=edges.device)
            row, col = edges[0], edges[1]
            idx = col
            deg = scatter_add(edge_weight, idx, dim=0, dim_size=x.size(0))
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            weight = edge_weight*weight

            weight = torch.ones_like(edges[0], dtype= torch.float)

            x = torch.relu(layer(x, edges, weight))

        edges = edge_index[0] if layerwise_adjacency else edge_index
        weight = torch.tensor(edge_weight[0]) if layerwise_adjacency else edge_weight
        weight = torch.ones_like(edges[0], dtype= torch.float)
        logits = self.gcn_layers[-1](x, edges, weight)

        return logits
