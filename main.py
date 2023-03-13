from sklearn.metrics import accuracy_score
from tap import Tap
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import wandb
import math

from modules.gcn import GCN
from modules.utils import get_neighboring_nodes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'
    num_hops: int = 2
    notes: str = None
    log_wandb: bool = False
    batch_size: int = 16


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)

    data = Planetoid(root='data/Planetoid', name=args.dataset)[0]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    num_classes = len(data.y.unique())

    model = GCN(data.num_features, hidden_dims=[32, num_classes]).to(device)
    optimizer = Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    train_idx = data.train_mask.nonzero()
    train_num_batches = max(math.ceil(len(train_idx)/args.batch_size), 1)
    batch_size = min(args.batch_size, len(data.train_mask))
    adjacency = torch.sparse_coo_tensor(indices=data.edge_index,
                                        values=torch.ones(data.edge_index.shape[-1]),
                                        size=(data.num_nodes, data.num_nodes))

    max_epochs = 100
    with tqdm(range(max_epochs)) as loop:
        for epoch in loop:
            for batch_id in range(train_num_batches):
                if batch_id == train_num_batches-1:
                    batch_nodes = train_idx[batch_id * batch_size:]
                else:
                    batch_nodes = train_idx[batch_id*batch_size:(batch_id+1)*batch_size]

                # Here's where we use GCN-GC to sample
                for hop in range(args.num_hops):
                    # Get neighborhoods of nodes in batch
                    neighborhoods = get_neighboring_nodes(batch_nodes, adjacency)

                    # Select only rows of feature matrices that we need

                    # Pass neighborhoods to GCN-GF

                    # Get probabilities for sampling each node

                    # Sample Ai using the probabilities

                    # Update batch_nodes

                # Pass A1, A2, ... to GCN-C

                logits = model(data.x, data.edge_index)
                loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = evaluate(logits, y, data.val_mask)
                wandb.log({'valid-accuracy': accuracy})
                wandb.log({'loss': loss.item()})

                loop.set_postfix({'loss': loss.item(), 'valid_acc': accuracy},
                                refresh=False)

    test_accuracy = evaluate(logits, y, data.test_mask)
    print(f'Test accuracy: {test_accuracy:.1%}')
    wandb.log({'test-accuracy': test_accuracy})


@torch.inference_mode()
def evaluate(logits: torch.Tensor,
             targets: torch.Tensor,
             mask: torch.Tensor
             ) -> float:
    predictions = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(predictions[mask].cpu(), targets[mask].cpu())
    return accuracy


train(Arguments().parse_args())
