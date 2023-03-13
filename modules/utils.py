import time
import os

import torch


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


def get_neighboring_nodes(nodes, adjecency_matrix):
    """ Returns a list of neighboring nodes for each node in `nodes """

    assert type(nodes) == torch.Tensor  # nodes should be a tensor
    assert type(adjecency_matrix) == torch.Tensor  # adjecency_matrix should be a tensor

    isin_row = torch.isin(adjecency_matrix._indices()[0], nodes)
    isin_col = torch.isin(adjecency_matrix._indices()[1], nodes)
    isin = isin_row | isin_col
    edge_index = adjecency_matrix._indices()[:, torch.where(isin)[0]]

    print('nodes', nodes)
    print('edge_index', edge_index)

    # Convert the list of neighboring nodes to a tensor
    return edge_index


# x = get_neighboring_nodes(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
# print(x)
