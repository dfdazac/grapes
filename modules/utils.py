import logging
from typing import Dict, Tuple

import scipy.sparse as sp
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Gumbel
import numpy as np
import os
import psutil


from modules.simple import KSubsetDistribution


def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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
            logits.squeeze(-1)).log(), {}
    assert k < n
    assert k > 0

    b = Bernoulli(logits=logits.squeeze())

    # Gumbel-sort trick https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    gumbel = Gumbel(torch.tensor(0., device=logits.device), torch.tensor(1., device=logits.device))
    gumbel_noise = gumbel.sample((n,))
    perturbed_log_probs = b.probs.log() + gumbel_noise

    samples = torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1]

    # entropy = b.entropy()
    # calculate the entropy in bits
    entropy = torch.tensor(-(b.probs * b.probs.log2() + (1 - b.probs) * (1 - b.probs).log2()))

    min_prob = b.probs.min(-1)[0]
    max_prob = b.probs.max(-1)[0]

    std_entropy, mean_entropy = torch.std_mean(entropy)

    mask = torch.zeros_like(logits.squeeze(), dtype=torch.float)
    mask[samples] = 1

    neighbor_nodes = neighbor_nodes[mask.bool().cpu()]

    stats_dict = {"min_prob": min_prob,
                  "max_prob": max_prob,
                  "mean_entropy": mean_entropy,
                  "std_entropy": std_entropy}

    return neighbor_nodes, b.log_prob(mask), stats_dict


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


# From PyGAS, PyTorch Geometric Auto-Scale: https://github.com/rusty1s/pyg_autoscale/tree/master
def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


# Function to return memory usage in MB
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # Convert bytes to MB

