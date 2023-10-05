import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
from torch.distributions import Binomial
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import Planetoid, Reddit
from tqdm import tqdm
from modules.utils import *

from modules.gcn import GCN
from modules.utils import (TensorMap, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency,
                           get_logger)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'reddit'

    sampling_hops: int = 2
    num_samples: int = 512
    constrain_k_weight: float = 0.001
    lr_gc: float = 1e-3
    sample_method: str = 'random'

    hidden_dim: int = 32
    max_epochs: int = 100
    batch_size: int = 512
    eval_frequency: int = 5
    eval_on_cpu: bool = False

    notes: str = None
    log_wandb: bool = True


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               entity='gflow-samp',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger()

    if args.dataset == 'reddit':
        path = os.path.join(os.getcwd(), 'data', 'Reddit')
        dataset = Reddit(path)
        data = dataset[0]
    else:
        data = Planetoid(root='data/Planetoid', name=args.dataset)[0]

    num_classes = len(data.y.unique())
    node_map = TensorMap(size=data.num_nodes)

    gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes]).to(device)
    optimizer_c = Adam(gcn_c.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    train_idx = data.train_mask.nonzero().squeeze(1)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)
    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))
    lap_matrix = row_normalize(adjacency + sp.eye(adjacency.shape[0]))

    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_c = 0
        with tqdm(total=len(loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(loader):
                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                global_edge_indices = []
                # Sample neighborhoods with the GCN-GF model
                for hop in range(args.sampling_hops):
                    # Get neighborhoods of target nodes in batch
                    neighborhoods = get_neighborhoods(previous_nodes, adjacency)

                    # Identify batch nodes (nodes + neighbors) and neighbors
                    prev_nodes_mask.zero_()
                    batch_nodes_mask.zero_()
                    prev_nodes_mask[previous_nodes] = True
                    batch_nodes_mask[neighborhoods.view(-1)] = True
                    neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

                    batch_nodes = node_map.values[batch_nodes_mask]
                    neighbor_nodes = node_map.values[neighbor_nodes_mask]

                    # Map neighborhoods to local node IDs
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)
                    # adj_row = lap_matrix[previous_nodes.squeeze(), :]
                    adj_row = adjacency[previous_nodes.squeeze(), :]
                    pi = np.array(np.sum(adj_row.multiply(adj_row), axis=0))[0]
                    p = pi / np.sum(pi)
                    s_num = np.min([np.sum(p > 0), args.num_samples])
                    if args.sample_method == 'ladies':
                        sampled_neighboring_nodes = np.random.choice(data.num_nodes, s_num, p=p, replace=False)
                    elif args.sample_method == 'random':
                        sampled_neighboring_nodes = torch.multinomial(
                             torch.ones_like(neighbor_nodes)/len(neighbor_nodes), s_num, replacement=False)
                    sampled_neighboring_nodes = torch.tensor(sampled_neighboring_nodes)

                    all_nodes_mask[sampled_neighboring_nodes] = True
                    # Update batch nodes
                    batch_nodes = torch.unique(torch.cat([target_nodes, sampled_neighboring_nodes], dim=0))

                    k_hop_edges = slice_adjacency(adjacency,
                                                  rows=previous_nodes,
                                                  cols=batch_nodes)
                    global_edge_indices.append(k_hop_edges)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]

                x = data.x[all_nodes].to(device)
                logits = gcn_c(x, edge_indices)

                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits[local_target_ids],
                                 data.y[target_nodes].to(device))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                batch_loss_c = loss_c.item()

                wandb.log({'batch_loss_c': batch_loss_c})

                acc_loss_c += batch_loss_c / len(loader)

                bar.set_postfix({'batch_loss_c': batch_loss_c})
                bar.update()

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:
            if args.eval_on_cpu:
                gcn_c.cpu()
            accuracy, f1 = evaluate(gcn_c,
                                    data,
                                    data.val_mask,
                                    args.eval_on_cpu)
            if args.eval_on_cpu:
                gcn_c.to(device)
            wandb.log({'epoch': epoch,
                       'loss_c': acc_loss_c,
                       'valid_accuracy': accuracy,
                       'valid_f1': f1})
            logger.info(f'loss_c={acc_loss_c:.6f}, '
                        f'valid_accuracy={accuracy:.3f}, '
                        f'valid_f1={f1:.3f}')

    test_accuracy, test_f1 = evaluate(gcn_c, data, data.test_mask, args.eval_on_cpu)
    print(f'Test accuracy: {test_accuracy:.1%}'
          f' Test f1: {test_f1:.1%}')
    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})


@torch.inference_mode()
def evaluate(model,
             data,
             mask: torch.Tensor,
             eval_on_cpu: bool
             ) -> tuple[float, float]:
    get_logger().info('Evaluating')

    x = data.x
    edge_index = data.edge_index
    if not eval_on_cpu:
        x = x.to(device)
        edge_index = edge_index.to(device)

    # perform full batch message passing for evaluation
    logits_total = model(x, edge_index)

    predictions = torch.argmax(logits_total, dim=1)[mask].cpu()
    targets = data.y[mask]
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='micro')

    return accuracy, f1


train(Arguments(explicit_bool=True).parse_args())
