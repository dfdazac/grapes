import os
import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.distributions import Bernoulli

from modules.simple import KSubsetDistribution


def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return neighbor_nodes, torch.sigmoid(logits.squeeze(-1)).log().sum()
    assert k < n
    assert k > 0
    sampling_rate = k / n
    logit_bias = -np.log((1 / sampling_rate) - 1)
    logit = logits.squeeze(-1) + logit_bias
    b = Bernoulli(logits=logit)
    samples = b.sample()
    neighbor_nodes = neighbor_nodes[(samples == 1).cpu()]
    return neighbor_nodes, b.log_prob(samples)


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


def get_neighboring_nodes(nodes: torch.Tensor,
                          adjacency: sp.csr_matrix
                          ) -> torch.Tensor:
    """Returns the neighbors of a set of nodes from a given adjacency matrix"""
    neighborhood = adjacency[nodes].tocoo()
    edge_index = torch.stack([nodes[neighborhood.row],
                              torch.tensor(neighborhood.col)],
                             dim=0)
    return edge_index
