from sklearn.metrics import accuracy_score
from tap import Tap
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import wandb
import math
from modules.simple import KSubsetDistribution

from modules.gcn import GCN
from modules.utils import get_neighboring_nodes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'
    num_hops: int = 2
    notes: str = None
    log_wandb: bool = False
    batch_size: int = 16
    num_sample: int = 20


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)

    data = Planetoid(root='data/Planetoid', name=args.dataset)[0]
    # edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    num_classes = len(data.y.unique())

    gcn_c = GCN(data.num_features, hidden_dims=[32, num_classes]).to(device)
    gcn_gf = GCN(data.num_features, hidden_dims=[32, 1]).to(device)
    optimizer_c = Adam(gcn_c.parameters(), lr=1e-2)
    optimizer_gf = Adam(gcn_gf.parameters(), lr=1e-2)
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
                    target_nodes = train_idx[batch_id * batch_size:]
                else:
                    target_nodes = train_idx[batch_id*batch_size:(batch_id+1)*batch_size]

                # Here's where we use GCN-GF to sample
                for hop in range(args.num_hops):

                    print('target_nodes', target_nodes.shape)
                    # A1 - Get 1-hop neighborhoods of nodes in batch
                    a1_neighborhoods = get_neighboring_nodes(target_nodes, adjacency)
                    print('neighborhoods', a1_neighborhoods.shape)

                    # Select only rows of feature matrices that we need
                    a1_batch_nodes = torch.unique(a1_neighborhoods)  # Contains target nodes and their one-hop neighbors
                    print('a1_batch_nodes', a1_batch_nodes)

                    a1_neighbors = a1_batch_nodes[~a1_batch_nodes.unsqueeze(1).eq(target_nodes.t()).any(1)]
                    print('one_hop_neighbors', a1_neighbors)

                    global_to_local_idx = {i.item(): j for j, i in enumerate(a1_batch_nodes)}
                    local_to_global_idx = {v: k for k, v in global_to_local_idx.items()}
                    x = data.x[a1_batch_nodes]

                    # Build edge index with local identifiers
                    a1_local_neighborhoods = torch.zeros_like(a1_neighborhoods)
                    a1_local_neighborhoods[0] = torch.tensor([global_to_local_idx[i.item()] for i in a1_neighborhoods[0]])
                    a1_local_neighborhoods[1] = torch.tensor([global_to_local_idx[i.item()] for i in a1_neighborhoods[1]])

                    # A1 consists of target nodes and their 1-hop neighbors.
                    # GF GCN returns a probability for each node in A1.
                    a1_nodes_logits = gcn_gf(x.to(device), a1_local_neighborhoods.to(device))
                    print('logits', a1_nodes_logits.shape)

                    # Filter out target nodes from logits
                    a1_nodes_idx = torch.tensor([global_to_local_idx[i.item()] for i in a1_neighbors])
                    a1_nodes_logits = a1_nodes_logits[a1_nodes_idx]

                    # Get probabilities for sampling each node
                    a1_nodes_probs = torch.sigmoid(a1_nodes_logits)

                    # Sample k-nodes using the probabilities from GF GCN
                    a1_k_subset = KSubsetDistribution(a1_nodes_probs, args.num_sample)
                    a1_samples = a1_k_subset.sample().long()
                    a1_samples = a1_neighbors[a1_samples == 1]

                    assert len(a1_samples) == args.num_sample

                    # Get 2-hop neighborhoods of nodes in batch
                    a2_neighborhoods = get_neighboring_nodes(a1_neighbors, adjacency)
                    print('a2_neighborhoods', a2_neighborhoods.shape)

                    ############################

                    # Select only rows of feature matrices that we need
                    a2_batch_nodes = torch.unique(a2_neighborhoods)  # Contains a1 nodes and their one-hop neighbors
                    print('a2_batch_nodes', a2_batch_nodes)

                    a2_neighbors = a2_batch_nodes[~a2_batch_nodes.unsqueeze(1).eq(a1_batch_nodes.t()).any(1)]
                    print('two_hop_neighbors', a2_neighbors)

                    global_to_local_idx = {i.item(): j for j, i in enumerate(a2_batch_nodes)}
                    local_to_global_idx = {v: k for k, v in global_to_local_idx.items()}
                    x = data.x[a2_batch_nodes]

                    # Build edge index with local identifiers
                    a2_local_neighborhoods = torch.zeros_like(a2_neighborhoods)
                    a2_local_neighborhoods[0] = torch.tensor(
                        [global_to_local_idx[i.item()] for i in a2_neighborhoods[0]])
                    a2_local_neighborhoods[1] = torch.tensor(
                        [global_to_local_idx[i.item()] for i in a2_neighborhoods[1]])

                    # A1 consists of target nodes and their 1-hop neighbors.
                    # GF GCN returns a probability for each node in A1.
                    a2_nodes_logits = gcn_gf(x.to(device), a2_local_neighborhoods.to(device))
                    print('logits', a2_nodes_logits.shape)

                    # Filter out target nodes from logits
                    a2_nodes_idx = torch.tensor([global_to_local_idx[i.item()] for i in a2_neighbors])
                    a2_nodes_logits = a2_nodes_logits[a2_nodes_idx]

                    # Get probabilities for sampling each node
                    a2_nodes_probs = torch.sigmoid(a2_nodes_logits)

                    # Sample k-nodes using the probabilities from GF GCN
                    a2_k_subset = KSubsetDistribution(a2_nodes_probs, args.num_sample)
                    a2_samples = a2_k_subset.sample().long()
                    a2_samples = a2_neighbors[a2_samples == 1]

                    assert len(a2_samples) == args.num_sample

                    print(a2_samples)
                    exit()

                    # Update batch_nodes

                    optimizer_gf.zero_grad()
                    loss.backward()
                    optimizer_gf.step()

                # Pass A1, A2, ... to GCN-C

                logits = gcn_c(data.x, data.edge_index)
                loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

                optimizer_c.zero_grad()
                loss.backward()
                optimizer_c.step()

                accuracy = evaluate(logits, y, data.val_mask)
                wandb.log({'valid-accuracy': accuracy})
                wandb.log({'loss': loss.item()})

                loop.set_postfix({'loss': loss.item(), 'valid_acc': accuracy}, refresh=False)

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
