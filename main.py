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
    lr_gf: float = 1e-3
    lr_gc: float = 1e-3

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

    gcn_c = GCN(data.num_features, hidden_dims=[32, num_classes]).to(device)
    gcn_gf = GCN(data.num_features, hidden_dims=[32, 1]).to(device)
    log_z = torch.tensor(100., requires_grad=True)
    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=args.lr_gf)
    loss_fn = nn.CrossEntropyLoss()

    train_idx = data.train_mask.nonzero().squeeze(1)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)
    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))

    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_gfn = 0
        acc_loss_c = 0
        with tqdm(total=len(loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(loader):
                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                global_edge_indices = []
                log_probs = []
                sampled_sizes = []
                neighborhood_sizes = []
                all_statistics = []
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
                    # Select only the needed rows from the feature matrix
                    x = data.x[batch_nodes].to(device)

                    # Get probabilities for sampling each node
                    node_logits = gcn_gf(x, local_neighborhoods)
                    # Select logits for neighbor nodes only
                    node_logits = node_logits[node_map.map(neighbor_nodes)]

                    # Sample neighbors using the logits
                    sampled_neighboring_nodes, log_prob, statistics = sample_neighborhoods_from_probs(
                        node_logits,
                        neighbor_nodes,
                        args.num_samples
                    )
                    all_nodes_mask[sampled_neighboring_nodes] = True

                    log_probs.append(log_prob)
                    sampled_sizes.append(sampled_neighboring_nodes.shape[0])
                    neighborhood_sizes.append(neighborhoods.shape[-1])
                    all_statistics.append(statistics)

                    # Update batch nodes for next hop
                    batch_nodes = torch.cat([target_nodes,
                                             sampled_neighboring_nodes],
                                            dim=0)

                    # Retrieve the edge index that results after sampling
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

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()
                for i in range(len(sampled_sizes)):
                    # Check if the sampled size is likely under a
                    # binomial distribution with probability n/k
                    binom = Binomial(total_count=torch.tensor(neighborhood_sizes[i], device=cost_gfn.device),
                                     probs=torch.tensor(args.num_samples / neighborhood_sizes[i], device=cost_gfn.device))
                    cost_gfn += -binom.log_prob(torch.tensor(sampled_sizes[i], device=cost_gfn.device))

                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + cost_gfn)**2
                loss_gfn.backward()
                optimizer_gf.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

                wandb.log({'batch_loss_gfn': batch_loss_gfn,
                           'batch_loss_c': batch_loss_c})

                acc_loss_gfn += batch_loss_gfn / len(loader)
                acc_loss_c += batch_loss_c / len(loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                 'batch_loss_c': batch_loss_c})
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

            log_dict = {'epoch': epoch,
                        'loss_gfn': acc_loss_gfn,
                        'loss_c': acc_loss_c,
                        'valid_accuracy': accuracy,
                        'valid_f1': f1}
            for i, statistics in enumerate(all_statistics):
                for key, value in statistics.items():
                    log_dict[f"{key}_{i}"] = value
            wandb.log(log_dict)

            logger.info(f'loss_gfn={acc_loss_gfn:.6f}, '
                        f'loss_c={acc_loss_c:.6f}, '
                        f'valid_accuracy={accuracy:.3f}, '
                        f'valid_f1={f1:.3f}')

    if args.eval_on_cpu:
        gcn_c.cpu()
    test_accuracy, test_f1 = evaluate(gcn_c,
                                      data,
                                      data.test_mask,
                                      args.eval_on_cpu)
    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')


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
