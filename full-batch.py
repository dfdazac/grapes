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
    lr_gc: float = 1e-3
    use_indicators: bool = True
    lr_gf: float = 1e-4
    loss_coef: float = 1e4
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.
    input_features: bool = False

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

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)

    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    if args.input_features:
        features = data.x
    else:
        features = nn.Parameter(torch.FloatTensor(data.num_nodes, data.num_features), requires_grad=True)
        nn.init.kaiming_normal_(features, mode='fan_in')

    train_idx = data.train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)

    val_idx = data.val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data.test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_gfn = 0
        acc_loss_c = 0
        acc_loss_binom = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            x = features.to(device)
            logits, gcn_mem_alloc = gcn_c(x, data.edge_index.to(device))
            loss_c = loss_fn(logits[data.train_mask], data.y[data.train_mask].to(device))

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            wandb.log({'loss_c': loss_c.item()})

            bar.set_postfix({'loss_c': loss_c.item()})
            bar.update()

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:
            val_predictions = torch.argmax(logits, dim=1)[data.val_mask].cpu()
            targets = data.y[data.val_mask]
            accuracy = accuracy_score(targets, val_predictions)
            f1 = f1_score(targets, val_predictions, average='micro')

            log_dict = {'epoch': epoch,
                        'valid_f1': f1}

            logger.info(f'loss_c={acc_loss_c:.6f}, '
                        f'valid_f1={f1:.3f}')
            wandb.log(log_dict)

    x = features.to(device)
    logits, gcn_mem_alloc = gcn_c(x, data.edge_index.to(device))
    test_predictions = torch.argmax(logits, dim=1)[data.test_mask].cpu()
    targets = data.y[data.test_mask]
    test_accuracy = accuracy_score(targets, test_predictions)
    test_f1 = f1_score(targets, test_predictions, average='micro')

    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')

    return test_f1


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs, 3)
for r in range(args.runs):
    test_f1= train(args)
    results[r, 0] = test_f1

print(f'Acc: {100 * results[:,0].mean():.2f} ± {100 * results[:,0].std():.2f}')
