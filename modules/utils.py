import logging
from typing import Dict, Tuple

import scipy.sparse as sp
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Gumbel
import numpy as np

from modules.simple import KSubsetDistribution


def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1,
                                    replacement: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove edges from an edge index, by removing nodes according to some
    probability.

    Uses Gumbel-max trick to sample from Bernoulli distribution. This is off-policy, since the original input
    distribution is a regular Bernoulli distribution.
    Args:
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """

    k = num_samples
    n = neighbor_nodes.shape[0]
    if k >= n:
        # TODO: Test this setting
        return neighbor_nodes, torch.sigmoid(
            logits.squeeze(-1)).log().sum(), torch.ones_like(neighbor_nodes), torch.ones_like(neighbor_nodes), {}
    assert k < n
    assert k > 0

    logits = logits.squeeze()
    q_node = torch.softmax(logits, -1)
    samples = torch.multinomial(q_node, k, replacement=replacement)

    entropy = -(q_node * torch.log(q_node)).sum(-1)
    min_prob = q_node.min(-1)[0]
    max_prob = q_node.max(-1)[0]

    # count nodes sampled more than once
    unique_nodes, counts = torch.unique(samples, return_counts=True)
    non_unique_nodes = unique_nodes[counts > 1]
    non_unique_counts = counts[counts > 1]

    mask = torch.zeros_like(logits.squeeze(), dtype=torch.float)
    mask[unique_nodes] = 1

    neighbor_nodes = neighbor_nodes[mask.bool().cpu()]

    stats_dict = {"min_prob": min_prob,
                  "max_prob": max_prob,
                  "entropy": entropy,}

    return neighbor_nodes, q_node[samples], non_unique_nodes, non_unique_counts, stats_dict


def sample_neighborhood_simple(probabilities: torch.Tensor,
                               neighbor_nodes: torch.Tensor,
                               num_samples: int = -1
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove edges from an edge index, by removing nodes according to some
    probability.
    Args:
        probabilities: tensor of shape (N,), where N all the number of unique
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """
    if num_samples > 0:
        node_k_subset = KSubsetDistribution(probabilities, num_samples)
        node_samples = node_k_subset.sample()
        neighbor_nodes = neighbor_nodes[node_samples.long() == 1]

        # Check that we have the right number of samples
        assert len(neighbor_nodes) == num_samples
        return neighbor_nodes, node_k_subset.log_prob(node_samples)
    else:
        return neighbor_nodes, None


def get_neighborhoods(nodes: Tensor,
                      adjacency: sp.csr_matrix
                      ) -> Tensor:
    """Returns the neighbors of a set of nodes from a given adjacency matrix"""
    neighborhood = adjacency[nodes].tocoo()
    neighborhoods = torch.stack([nodes[neighborhood.row],
                                 torch.tensor(neighborhood.col)],
                                dim=0)
    return neighborhoods


def slice_adjacency(adjacency: sp.csr_matrix, rows: Tensor, cols: Tensor):
    """Selects a block from a sparse adjacency matrix, given the row and column
    indices. The result is returned as an edge index.
    """
    row_slice = adjacency[rows]
    row_col_slice = row_slice[:, cols]
    slice = row_col_slice.tocoo()
    edge_index = torch.stack([rows[slice.row],
                              cols[slice.col]],
                             dim=0)
    return edge_index


class TensorMap:
    """A class used to quickly map integers in a tensor to an interval of
    integers from 0 to len(tensor) - 1. This is useful for global to local
    conversions.

    Example:
        >>> nodes = torch.tensor([22, 32, 42, 52])
        >>> node_map = TensorMap(size=nodes.max() + 1)
        >>> node_map.update(nodes)
        >>> node_map.map(torch.tensor([52, 42, 32, 22, 22]))
        tensor([3, 2, 1, 0, 0])
    """

    def __init__(self, size):
        self.map_tensor = torch.empty(size, dtype=torch.long)
        self.values = torch.arange(size)

    def update(self, keys: Tensor):
        values = self.values[:len(keys)]
        self.map_tensor[keys] = values

    def map(self, keys):
        return self.map_tensor[keys]


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger('')
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx
