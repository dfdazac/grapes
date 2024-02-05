import os
import pdb

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from tap import Tap
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eval import evaluate
from modules.data import get_data
from modules.gcn import GCN
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency)
from torch_geometric.utils import homophily


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'

    sampling_hops: int = 2
    lr_gc: float = 1e-2
    dropout: float = 0.
    input_features: bool = False

    model_type: str = 'gcn'
    hidden_dim: int = 256
    max_epochs: int = 30
    batch_size: int = 512
    eval_frequency: int = 5
    eval_on_cpu: bool = True
    eval_full_batch: bool = True

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

    node_map = TensorMap(size=data.num_nodes)

    gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    if args.input_features:
        features = data.x
        optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    else:
        features = torch.FloatTensor(data.num_nodes, data.num_features)
        nn.init.kaiming_normal_(features, mode='fan_in')
        features = nn.Parameter(features, requires_grad=True)

        # Create optimizer
        optimizer_c = Adam(list(gcn_c.parameters())+[features], lr=args.lr_gc)

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

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_c = 0
        homophily_hop1 = 0
        homophily_hop2 = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                local_edge_indices = []
                neighborhood_sizes = []
                sub_adj_size = []
                # Sample neighborhoods with the GCN-GF model
                for hop in range(args.sampling_hops):
                    local_hop_edges = []
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

                    # Retrieve the edge index that results after sampling
                    k_hop_edges = slice_adjacency(adjacency,
                                                  rows=previous_nodes,
                                                  cols=batch_nodes)
                    k_hop_edges_w_sloop = torch.cat([k_hop_edges, target_nodes.repeat(2,1)], dim=1)
                    # import pdb; pdb.set_trace()

                    all_nodes_mask[batch_nodes] = True

                    sub_adj_size.append((len(previous_nodes), len(batch_nodes)))
                    neighborhood_sizes.append(neighborhoods.shape[-1])

                    node_map.update(previous_nodes)
                    local_hop_edges.append(node_map.map(k_hop_edges_w_sloop[0]))
                    node_map.update(batch_nodes)
                    local_hop_edges.append(node_map.map(k_hop_edges_w_sloop[1]))
                    local_edge_indices.append(local_hop_edges)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)


                batch_homophily_hop1 = homophily(local_edge_indices[0], data.y[all_nodes])
                batch_homophily_hop2 = homophily(local_edge_indices[1], data.y[all_nodes])

                x = features[batch_nodes].to(device)
                logits, gcn_mem_alloc = gcn_c(x, local_edge_indices, sub_adj_size)

                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits, data.y[target_nodes].to(device))
                print("features for all x before:", features[all_nodes])
                feature1 = features[all_nodes]
                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                print("features for all x after:", features[all_nodes])
                feature2 = features[all_nodes]
                batch_loss_c = loss_c.item()

                wandb.log({'batch_loss_c': batch_loss_c})

                acc_loss_c += batch_loss_c / len(train_loader)

                homophily_hop1 += batch_homophily_hop1 / len(train_loader)
                homophily_hop2 += batch_homophily_hop2 / len(train_loader)

                bar.set_postfix({'batch_loss_c': batch_loss_c})
                bar.update()

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:
            accuracy, f1 = evaluate(features,
                                    gcn_c,
                                    gcn_c,
                                    data,
                                    args,
                                    adjacency,
                                    node_map,
                                    0,
                                    device,
                                    data.val_mask,
                                    args.eval_on_cpu,
                                    loader=val_loader,
                                    full_batch=args.eval_full_batch,
                                    )
            if args.eval_on_cpu:
                gcn_c.to(device)

            log_dict = {'epoch': epoch,
                        'loss_c': acc_loss_c,
                        'valid_accuracy': accuracy,
                        'valid_f1': f1,
                        'homophily1': homophily_hop1,
                        'homophily2': homophily_hop2}

            print("homophily1:", homophily_hop1)
            print("homophily2:", homophily_hop2)


            logger.info(f'loss_c={acc_loss_c:.6f}, '
                        f'valid_accuracy={accuracy:.3f}, '
                        f'valid_f1={f1:.3f}')
            wandb.log(log_dict)

    test_accuracy, test_f1 = evaluate(features,
                                      gcn_c,
                                      gcn_c,
                                      data,
                                      args,
                                      adjacency,
                                      node_map,
                                      0,
                                      device,
                                      data.test_mask,
                                      args.eval_on_cpu,
                                      loader=test_loader,
                                      full_batch=args.eval_full_batch)
    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')
    return test_f1, 0,0,0


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs)
mem1 = []
mem2 = []
mem3 = []
for r in range(args.runs):
    test_f1, mean_mem1, mean_mem2, mean_mem3 = train(args)
    results[r] = test_f1

print(f'Memory point 1: {np.mean(mem1)} MB ± {np.std(mem1):.2f}')
print(f'Memory point 2: {np.mean(mem2)} MB ± {np.std(mem2):.2f}')
print(f'Memory point 2: {np.mean(mem3)} MB ± {np.std(mem3):.2f}')
print(f'Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')
