import pdb

from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
import torch
import scipy.sparse as sp
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from tqdm import tqdm
import wandb
import math
import os
from torch.distributions import Binomial
import time
from scipy.sparse import coo_matrix

from modules.gcn import GCN
from modules.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'reddit'
    num_hops: int = 2
    max_epochs: int = 100
    notes: str = None
    log_wandb: bool = True
    batch_size: int = 512
    num_samples: int = 512
    constrain_k_weight: float = 0.001


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               entity='gflow-samp',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)

    if args.dataset == 'reddit':
        path = os.path.join(os.getcwd(), 'data', 'Reddit')
        dataset = Reddit(path)
        data = dataset[0]
    else:
        data = Planetoid(root='data/Planetoid', name=args.dataset)[0]
    y = data.y.to(device)
    num_classes = len(data.y.unique())

    gcn_c = GCN(data.num_features, hidden_dims=[32, num_classes]).to(device)
    gcn_gf = GCN(data.num_features, hidden_dims=[32, 1]).to(device)
    log_z = torch.tensor(100., requires_grad=True)
    optimizer_c = Adam(gcn_c.parameters(), lr=1e-2)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    # pdb.set_trace()
    train_idx = data.train_mask.nonzero()
    train_num_batches = max(math.ceil(len(train_idx) / args.batch_size), 1)
    batch_size = min(args.batch_size, len(data.train_mask))
    # adjacency = get_adj(data.edge_index, data.num_nodes)
    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))
    lap_matrix = row_normalize(adjacency + sp.eye(adjacency.shape[0]))

    # pdb.set_trace()
    with tqdm(range(args.max_epochs)) as loop:
        for epoch in loop:
            for batch_id in range(train_num_batches):
                if batch_id == train_num_batches - 1:
                    target_nodes = train_idx[batch_id * batch_size:]
                else:
                    target_nodes = train_idx[batch_id * batch_size:(batch_id + 1) * batch_size]

                previous_nodes = target_nodes.clone()
                all_nodes = target_nodes.clone().squeeze(1)

                # Here's where we use GCN-GF to sample
                global_edge_indices = []
                log_probs = []
                sampled_sizes = []
                neighborhood_sizes = []
                for hop in range(args.num_hops):
                    # Get neighborhoods of target nodes in batch
                    # neighborhoods = get_neighboring_nodes(previous_nodes, adjacency)
                    # pdb.set_trace()
                    # adj_row = lap_matrix[previous_nodes.squeeze(), :]
                    adj_row = adjacency[previous_nodes.squeeze(), :]
                    pi = np.array(np.sum(adj_row.multiply(adj_row), axis=0))[0]
                    p = pi / np.sum(pi)
                    s_num = np.min([np.sum(p > 0), args.num_samples])
                    sampled_neighboring_nodes = np.random.choice(data.num_nodes, s_num, p=p, replace=False)
                    sampled_neighboring_nodes = torch.tensor(sampled_neighboring_nodes)

                    # Update batch nodes
                    batch_nodes = torch.unique(torch.cat([target_nodes.squeeze(1), sampled_neighboring_nodes], dim=0))
                    all_nodes = torch.unique(torch.cat([all_nodes, sampled_neighboring_nodes], dim=0))

                    # TODO Keep track of edges
                    row_isin = torch.isin(data.edge_index[0], previous_nodes)
                    col_isin = torch.isin(data.edge_index[1], batch_nodes)
                    isin = torch.logical_and(row_isin, col_isin)
                    edge_index_hop = data.edge_index[:, torch.where(isin)[0]]
                    global_edge_indices.append(edge_index_hop)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Pass A1, A2, ... (edge_index1, edge_index2, ...) to GCN-C

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second hop concatenated with the target nodes
                global_to_local_idx = {i.item(): j for j, i in enumerate(all_nodes)}
                local_edge_indices = []
                for edge_index in global_edge_indices:
                    local_index = torch.zeros_like(edge_index, device=device)
                    local_index[0] = torch.tensor([global_to_local_idx[i.item()] for i in edge_index[0]])
                    local_index[1] = torch.tensor([global_to_local_idx[i.item()] for i in edge_index[1]])
                    local_edge_indices.append(local_index)

                start_t = time.time()
                logits = gcn_c(data.x[all_nodes].to(device), local_edge_indices)
                local_target_ids = torch.tensor([global_to_local_idx[i.item()] for i in target_nodes])
                loss_c = loss_fn(logits[local_target_ids], data.y[target_nodes].squeeze().to(device))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                print(f'training time: {time.time() - start_t}')

                # print("Classification loss", loss_c, "GFN loss", loss_gfn)
                accuracy, f1 = evaluate(gcn_c, data, y, data.val_mask)
                wandb.log({
                    'epoch': epoch,
                    'loss_c': loss_c.item() / len(target_nodes),
                    'valid-accuracy': accuracy,
                    'valid-f1': f1
                })

                # # Update progress bar
                # loop.set_postfix({'loss': loss_c.item(),
                #                   'valid_acc': accuracy,
                #                  refresh=True)

    test_accuracy, test_f1 = evaluate(gcn_c, data, y, data.test_mask)
    print(f'Test accuracy: {test_accuracy:.1%}'
          f' Test f1: {test_f1:.1%}')
    wandb.log({'test-accuracy': test_accuracy,
               'test-f1': test_f1})


@torch.inference_mode()
def evaluate(model,
             data,
             targets: torch.Tensor,
             mask: torch.Tensor
             ) -> tuple[float, float]:
    # perform full batch message passing for evaluation
    logits_total = model(data.x.to(device), data.edge_index.to(device))

    predictions = torch.argmax(logits_total, dim=1)[mask].cpu()
    targets = targets[mask].cpu()
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='micro')

    return accuracy, f1


train(Arguments(explicit_bool=True).parse_args())
