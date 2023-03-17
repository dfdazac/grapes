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
from modules.utils import get_neighboring_nodes, sample_neighborhoods_from_probs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'
    num_hops: int = 2
    notes: str = None
    log_wandb: bool = False
    batch_size: int = 16
    num_samples: int = 20  # TODO Change this


def train(args: Arguments):
    wandb.init(project='gflow-sampling',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)

    data = Planetoid(root='data/Planetoid', name=args.dataset)[0]
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    num_classes = len(data.y.unique())

    gcn_c = GCN(data.num_features, hidden_dims=[32, num_classes]).to(device)
    gcn_gf = GCN(data.num_features, hidden_dims=[32, 1]).to(device)
    log_z = torch.tensor(10., requires_grad=True).to(device)
    optimizer_c = Adam(gcn_c.parameters(), lr=1e-2)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    train_idx = data.train_mask.nonzero()
    train_num_batches = max(math.ceil(len(train_idx) / args.batch_size), 1)
    batch_size = min(args.batch_size, len(data.train_mask))
    adjacency = torch.sparse_coo_tensor(indices=data.edge_index,
                                        values=torch.ones(data.edge_index.shape[-1]),
                                        size=(data.num_nodes, data.num_nodes))

    max_epochs = 100
    with tqdm(range(max_epochs)) as loop:
        for epoch in loop:
            for batch_id in range(train_num_batches):
                if batch_id == train_num_batches - 1:
                    target_nodes = train_idx[batch_id * batch_size:]
                else:
                    target_nodes = train_idx[batch_id * batch_size:(batch_id + 1) * batch_size]

                previous_nodes = target_nodes.clone()

                # Here's where we use GCN-GF to sample
                global_edge_indices = []
                log_probs = []
                for hop in range(args.num_hops):
                    # Get neighborhoods of target nodes in batch
                    neighborhoods = get_neighboring_nodes(previous_nodes, adjacency)

                    # Select only rows of feature matrices that we need
                    batch_nodes = torch.unique(neighborhoods)  # Contains target nodes and their one-hop neighbors
                    neighbor_nodes = batch_nodes[~torch.isin(batch_nodes, previous_nodes)]

                    global_to_local_idx = {i.item(): j for j, i in enumerate(batch_nodes)}
                    x = data.x[batch_nodes]

                    # Build edge index with local identifiers
                    local_neighborhoods = torch.zeros_like(neighborhoods)
                    local_neighborhoods[0] = torch.tensor([global_to_local_idx[i.item()] for i in neighborhoods[0]])
                    local_neighborhoods[1] = torch.tensor([global_to_local_idx[i.item()] for i in neighborhoods[1]])

                    # Pass neighborhoods to GCN-GF and get probabilities for sampling each node
                    node_logits = gcn_gf(x.to(device), local_neighborhoods.to(device))
                    node_probs = torch.sigmoid(node_logits)

                    # Filter out probabilities of target nodes
                    nodes_idx = torch.tensor([global_to_local_idx[i.item()] for i in neighbor_nodes])
                    node_probs = node_probs[nodes_idx]

                    # Sample Ai using the probabilities
                    sampled_neighboring_nodes, log_prob = sample_neighborhoods_from_probs(
                        node_probs,
                        neighbor_nodes,
                        args.num_samples
                    )

                    log_probs.append(log_prob)

                    # Update batch nodes
                    batch_nodes = torch.unique(sampled_neighboring_nodes)

                    # TODO Keep track of edges
                    row_isin = torch.isin(data.edge_index[0], previous_nodes)
                    col_isin = torch.isin(data.edge_index[1], batch_nodes)
                    isin = torch.logical_and(row_isin, col_isin)
                    edge_index_hop = edge_index[:, torch.where(isin)[0]]
                    global_edge_indices.append(edge_index_hop)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Pass A1, A2, ... (edge_index1, edge_index2, ...) to GCN-C

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second hop concatenated with the target nodes
                global_to_local_idx = {i.item(): j for j, i in enumerate(batch_nodes)}
                local_edge_indices = []
                for edge_index in global_edge_indices:
                    local_index = torch.zeros_like(edge_index, device=device)
                    local_index[0] = torch.tensor([global_to_local_idx[i.item()] for i in edge_index[0]])
                    local_index[1] = torch.tensor([global_to_local_idx[i.item()] for i in edge_index[1]])
                    local_edge_indices.append(local_index)

                logits = gcn_c(data.x[batch_nodes].to(device),
                               local_edge_indices)
                local_target_ids = torch.tensor([global_to_local_idx[i.item()] for i in target_nodes])
                loss_c = loss_fn(logits[local_target_ids],
                               data.y[target_nodes].squeeze().to(device))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                optimizer_gf.zero_grad()
                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + loss_c.detach())**2
                loss_gfn.backward()
                optimizer_gf.step()

                accuracy = evaluate(gcn_c, data, y, data.val_mask)
                wandb.log({'valid-accuracy': accuracy})
                wandb.log({'loss': loss_c.item()})

                loop.set_postfix({'loss': loss_c.item(), 'valid_acc': accuracy},
                                 refresh=False)

    test_accuracy = evaluate(gcn_c, data, y, data.test_mask)
    print(f'Test accuracy: {test_accuracy:.1%}')
    wandb.log({'test-accuracy': test_accuracy})


@torch.inference_mode()
def evaluate(model,
             data,
             targets: torch.Tensor,
             mask: torch.Tensor
             ) -> float:
    # perform full batch message passing for evaluation
    logits_total = model(data.x.to(device), data.edge_index)

    predictions = torch.argmax(logits_total, dim=1)
    accuracy = accuracy_score(predictions[mask].cpu(), targets[mask].cpu())
    return accuracy


train(Arguments().parse_args())