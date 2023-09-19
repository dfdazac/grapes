import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch_geometric
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modules.data import get_data, get_ppi
from modules.gcn import GCN, GCN2
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'

    sampling_hops: int = 2
    num_samples: int = 16
    use_indicators: bool = True
    lr_gf: float = 1e-4
    lr_gc: float = 1e-3
    loss_coef: float = 1e4
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.

    model_type: str = 'gcn'
    hidden_dim: int = 256
    max_epochs: int = 30
    batch_size: int = 512
    eval_frequency: int = 5
    eval_on_cpu: bool = False
    eval_full_batch: bool = False

    runs: int = 10
    notes: str = None
    log_wandb: bool = True
    config_file: str = None


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               entity='gflow-samp',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger()

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset)

    if args.dataset == 'ppi':
        # PPI is inductive, it has separate datasets for each split.
        val_data, _, _ = get_ppi(path, split='val')
        test_data, _, _ = get_ppi(path, split='test')

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn2':
        # GCN model  for classification
        gcn_c = GCN2(data.num_features, hidden_dims=[args.hidden_dim, num_classes], alpha=0.1, theta=0.5).to(device)
        # GCN model for GFlotNet sampling
        gcn_gf = GCN2(data.num_features + num_indicators,
                    hidden_dims=[args.hidden_dim, 1], alpha=0.1, theta=0.5).to(device)
    elif args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
        # GCN model for GFlotNet sampling
        gcn_gf = GCN(data.num_features + num_indicators,
                      hidden_dims=[args.hidden_dim, 1]).to(device)

    log_z = torch.tensor(args.log_z_init, requires_grad=True)
    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=args.lr_gf)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_idx = data.train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)

    val_idx = data.val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data.test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

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
        acc_loss_binom = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                indicator_features.zero_()
                indicator_features[target_nodes, -1] = 1.0

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
                                 data.y[target_nodes].to(device)) + args.reg_param*torch.sum(torch.var(logits, dim=1))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()

                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + args.loss_coef*cost_gfn)**2
                loss_gfn.backward()
                optimizer_gf.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

                wandb.log({'batch_loss_gfn': batch_loss_gfn,
                           'batch_loss_c': batch_loss_c,
                           'log_z': log_z,
                           '-log_probs': -torch.sum(torch.cat(log_probs, dim=0))})

                acc_loss_gfn += batch_loss_gfn / len(train_loader)
                acc_loss_c += batch_loss_c / len(train_loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                 'batch_loss_c': batch_loss_c,
                                 'log_z': log_z.item(),
                                 'log_probs': torch.sum(torch.cat(log_probs, dim=0)).item()})
                bar.update()

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:
            accuracy, f1 = evaluate(gcn_c,
                                    gcn_gf,
                                    data,
                                    args,
                                    adjacency,
                                    node_map,
                                    num_indicators,
                                    data.val_mask,
                                    args.eval_on_cpu,
                                    loader=val_loader,
                                    full_batch=args.eval_full_batch,
                                    )
            if args.eval_on_cpu:
                gcn_c.to(device)

            log_dict = {'epoch': epoch,
                        'loss_gfn': acc_loss_gfn,
                        'loss_c': acc_loss_c,
                        'loss_binom': acc_loss_binom,
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

    test_accuracy, test_f1 = evaluate(gcn_c,
                                      gcn_gf,
                                      data,
                                      args,
                                      adjacency,
                                      node_map,
                                      num_indicators,
                                      data.test_mask,
                                      args.eval_on_cpu,
                                      loader=test_loader,
                                      full_batch=args.eval_full_batch)
    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')
    return test_f1


@torch.inference_mode()
def evaluate(gcn_c: torch.nn.Module,
             gcn_gf: torch.nn.Module,
             data: torch_geometric.data.Data,
             args: argparse.Namespace,
             adjacency: torch.Tensor,
             node_map: TensorMap,
             num_indicators: int,
             mask: torch.Tensor = None,
             eval_on_cpu: bool = True,
             loader: torch.utils.data.DataLoader = None,
             full_batch: bool = False
             ) -> tuple[float, float]:
    """
    Evaluate the model on the validation or test set. This can be done in two ways: either by performing full-batch
    message passing or by performing mini-batch message passing. The latter is more memory efficient, but the former is
    faster.
    """
    get_logger().info('Evaluating')

    x = data.x
    edge_index = data.edge_index
    if eval_on_cpu:
        # move data to CPU
        x = x.cpu()
        edge_index = edge_index.cpu()
        gcn_c = gcn_c.cpu()
        all_predictions = torch.tensor([], dtype=torch.long, device='cpu')
    else:
        # move data to GPU
        x = x.to(device)
        edge_index = edge_index.to(device)
        gcn_c = gcn_c.to(device)
        all_predictions = torch.tensor([], dtype=torch.long, device='cuda')

    if full_batch:
        # perform full batch message passing for evaluation

        logits_total = gcn_c(x, edge_index)
        if data.y[mask].dim() == 1:
            predictions = torch.argmax(logits_total, dim=1)[mask].cpu()
            targets = data.y[mask]
            accuracy = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions, average='micro')
        # multilabel classification
        else:
            y_pred = logits_total[mask] > 0
            y_true = data.y[mask] > 0.5

            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = accuracy = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.
    else:
        # perform mini-batch message passing for evaluation
        assert loader is not None, 'loader must be provided if full_batch is False'

        prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        indicator_features = torch.zeros((data.num_nodes, num_indicators))

        for batch_id, batch in enumerate(loader):
            target_nodes = batch[0]

            previous_nodes = target_nodes.clone()
            all_nodes_mask = torch.zeros_like(prev_nodes_mask)
            all_nodes_mask[target_nodes] = True

            indicator_features.zero_()
            indicator_features[target_nodes, -1] = 1.0

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
                node_logits = gcn_gf(x, local_neighborhoods)
                # Select logits for neighbor nodes only
                node_logits = node_logits[node_map.map(neighbor_nodes)]

                # Sample top k neighbors using the logits
                b = Bernoulli(logits=node_logits.squeeze())
                samples = torch.topk(b.probs, k=args.num_samples, dim=0, sorted=False)[1]
                sample_mask = torch.zeros_like(node_logits.squeeze(), dtype=torch.float)
                sample_mask[samples] = 1
                sampled_neighboring_nodes = neighbor_nodes[sample_mask.bool().cpu()]

                all_nodes_mask[sampled_neighboring_nodes] = True

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

            all_nodes = node_map.values[all_nodes_mask]
            node_map.update(all_nodes)
            edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]

            x = data.x[all_nodes].to(device)
            logits_total = gcn_c(x, edge_indices)
            predictions = torch.argmax(logits_total, dim=1)
            predictions = predictions[node_map.map(target_nodes)]  # map back to original node IDs

            all_predictions = torch.cat([all_predictions, predictions], dim=0)

        all_predictions = all_predictions.cpu()
        targets = data.y[mask]

        accuracy = accuracy_score(targets, all_predictions)
        f1 = f1_score(targets, all_predictions, average='micro')

    return accuracy, f1


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs)
for r in range(args.runs):
    test_f1 = train(args)
    results[r] = test_f1

print(f'Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')
