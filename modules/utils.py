import torch


def get_neighboring_nodes(nodes, adjecency_matrix):
    """ Returns a list of neighboring nodes for each node in `nodes """

    assert type(nodes) == torch.Tensor  # nodes should be a tensor
    assert type(adjecency_matrix) == torch.Tensor  # adjecency_matrix should be a tensor

    # Get the neighboring nodes for each node in `nodes`
    return [adjecency_matrix[node, :].nonzero().squeeze() for node in nodes]


x = get_neighboring_nodes(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
print(x)
