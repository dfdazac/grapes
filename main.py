import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
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
    sample_with_replacement: bool = True
    use_indicators: bool = True
    lr_gf: float = 1e-4
    lr_gc: float = 1e-3
    loss_coef: float = 1e4
    log_z_init: float = 0.
    multi_sampled_weight: bool = False

    hidden_dim: int = 256
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

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0
    gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], normalize=False).to(device)
    gcn_gf = GCN(data.num_features + num_indicators,
                 hidden_dims=[args.hidden_dim, 1], normalize=False).to(device)
    log_z = torch.tensor(args.log_z_init, requires_grad=True)
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
    indicator_features = torch.zeros((data.num_nodes, num_indicators))

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

                indicator_features.zero_()
                indicator_features[target_nodes, -1] = 1.0

                global_edge_indices = []
                edge_weights = []
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
                    indicator_features[neighbor_nodes, hop] = 1.0

                    # Map neighborhoods to local node IDs
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)
                    # Select only the needed rows from the feature and
                    # indicator matrices
                    if args.use_indicators:
                        x = torch.cat([data.x[batch_nodes],
                                       indicator_features[batch_nodes]],
                                      dim=1
                                      ).to(device)
                    else:
                        x = data.x[batch_nodes].to(device)

                    # Get probabilities for sampling each node
                    gcn_gf_edge_weights = torch.tensor(torch.ones_like(local_neighborhoods[1], dtype=torch.float32))
                    node_logits = gcn_gf(x, local_neighborhoods, gcn_gf_edge_weights)
                    # get all nodes' probability of being sampled
                    node_logits = node_logits.squeeze()
                    node_probs = torch.softmax(node_logits, -1)
                    # get probs for neighbor nodes and target nodes separately
                    neighbor_probs = node_probs[node_map.map(neighbor_nodes)]
                    target_probs = node_probs[node_map.map(target_nodes)]

                    # Sample neighbors using the logits
                    sampled_neighboring_nodes, sampled_probs, non_unique_nodes, non_unique_counts, statistics = \
                        sample_neighborhoods_from_probs(
                            node_probs,
                            neighbor_probs,
                            neighbor_nodes,
                            args.num_samples,
                            args.sample_with_replacement
                        )
                    all_nodes_mask[sampled_neighboring_nodes] = True

                    log_probs.append(sampled_probs.log())

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

                    # get node weight of the nodes samples more than once

                    multi_sampled_node_weight = torch.ones_like(neighbor_nodes)
                    multi_sampled_node_weight[non_unique_nodes] = non_unique_counts.cpu()

                    node_map.update(neighbor_nodes)
                    # node_probs = torch.softmax(node_logits[node_map.map(sampled_neighboring_nodes)], 0)

                    node_weight_temp = torch.cat([torch.ones_like(target_nodes)/(target_probs),
                                                  multi_sampled_node_weight[node_map.map(sampled_neighboring_nodes)]
                                                  / (neighbor_probs[node_map.map(
                                                      sampled_neighboring_nodes)].to('cpu').squeeze())])


                    # version 1
                    # node_weights_dict = {k.item(): v for k, v in zip(batch_nodes, node_weight_temp.detach())}
                    # edge_weights.append(torch.tensor([node_weights_dict[i.item()] for i in k_hop_edges[1]]))
                    # version 2
                    node_weights = torch.zeros(batch_nodes.max()+1)
                    # node_weights[batch_nodes] = node_weight_temp
                    # edge_weights.append(node_weights[k_hop_edges[1]])
                    # version 3
                    node_weights1 = node_weights.scatter_(0, batch_nodes, node_weight_temp)
                    edge_weights.append(torch.gather(node_weights1, 0, k_hop_edges[1]))
                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                edge_weights_dev = [w.to(device) for w in edge_weights]

                edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
                x = data.x[all_nodes].to(device)
                logits = gcn_c(x, edge_indices, edge_weights_dev)

                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits[local_target_ids],
                                 data.y[target_nodes].to(device))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()

                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + args.loss_coef * cost_gfn) ** 2
                loss_gfn.backward()
                optimizer_gf.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

                wandb.log({'batch_loss_gfn': batch_loss_gfn,
                           'batch_loss_c': batch_loss_c,
                           'log_z': log_z,
                           '-log_probs': -torch.sum(torch.cat(log_probs, dim=0))})

                acc_loss_gfn += batch_loss_gfn / len(loader)
                acc_loss_c += batch_loss_c / len(loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                 'batch_loss_c': batch_loss_c,
                                 'log_z': log_z,
                                 'log_probs': torch.sum(torch.cat(log_probs, dim=0))})
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
    eval_edge_weights = torch.tensor(torch.ones_like(edge_index[1], dtype=torch.float32))
    logits_total = model(x, edge_index, eval_edge_weights)

    predictions = torch.argmax(logits_total, dim=1)[mask].cpu()
    targets = data.y[mask]
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='micro')

    return accuracy, f1


train(Arguments(explicit_bool=True).parse_args())
