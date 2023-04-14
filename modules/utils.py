import logging
import os
import time
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.distributions import Bernoulli

from modules.simple import KSubsetDistribution


def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1
                                    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove edges from an edge index, by removing nodes according to some
    probability.
    Args:
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """

    k = num_samples
    n = neighbor_nodes.shape[0]
    if k == n:
        return neighbor_nodes, torch.sigmoid(logits.squeeze(-1)).log().sum(), {}
    assert k < n
    assert k > 0
    sampling_rate = k / n
    logit_bias = -np.log((1 / sampling_rate) - 1)
    logit = logits.squeeze(-1) + logit_bias

    b = Bernoulli(logits=logit)
    entropy = b.entropy()
    min_prob = b.probs.min(-1)[0]
    max_prob = b.probs.max(-1)[0]

    mean_entropy = entropy.mean()
    var_entropy = torch.std(entropy)

    samples = b.sample()
    k_sampled = samples.sum()
    neighbor_nodes = neighbor_nodes[(samples == 1).cpu()]
    return neighbor_nodes, b.log_prob(samples), {"min_prob": min_prob, "max_prob": max_prob, "mean_entropy": mean_entropy, "var_entropy": var_entropy, "k_nodes_sampled": k_sampled}


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


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))


def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)


tics = []


def tic():
    tics.append(time.time())


def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()


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
