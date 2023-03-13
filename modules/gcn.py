import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int]):
        super(GCN, self).__init__()

        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                ) -> torch.Tensor:
        for layer in self.gcn_layers[:-1]:
            x = torch.relu(layer(x, edge_index))
        logits = self.gcn_layers[-1](x, edge_index)
        return logits
