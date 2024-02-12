import argparse

import torch
import torch_geometric
from sklearn.metrics import accuracy_score, f1_score
from torch.distributions import Bernoulli
import numpy as np

from modules.utils import TensorMap, get_logger, get_neighborhoods, slice_adjacency


@torch.inference_mode()
def evaluate(features: torch.Tensor,
             gcn_c: torch.nn.Module,
             gcn_gf: torch.nn.Module,
             data: torch_geometric.data.Data,
             args: argparse.Namespace,
             adjacency: torch.Tensor,
             node_map: TensorMap,
             num_indicators: int,
             device: torch.device,
             mask: torch.Tensor = None,
             eval_on_cpu: bool = True,
             loader: torch.utils.data.DataLoader = None,
             full_batch: bool = False,
             ) -> tuple[float, float]:
    """
    Evaluate the model on the validation or test set. This can be done in two ways: either by performing full-batch
    message passing or by performing mini-batch message passing. The latter is more memory efficient, but the former is
    faster.
    """
    get_logger().info('Evaluating')

    x = features
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
        edge_index = torch.cat([edge_index, torch.arange(data.num_nodes).repeat(2, 1)], dim=1)
        logits_total, _ = gcn_c(x, [[edge_index[0], edge_index[1]], [edge_index[0], edge_index[1]]],
                                [(data.num_nodes,data.num_nodes),(data.num_nodes,data.num_nodes)])
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
            local_edge_indices = []
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
                indicator_features[neighbor_nodes, hop] = 1.0

                # neighborhoods_inv = slice_adjacency(adjacency, batch_nodes, batch_nodes)
                # neighborhoods_inv_sloop = torch.cat([neighborhoods_inv, batch_nodes.repeat(2, 1)], dim=1)
                # Map neighborhoods to local node IDs
                node_map.update(batch_nodes)
                # local_neighborhoods = node_map.map(neighborhoods_inv_sloop).to(device)
                local_neighborhoods = node_map.map(neighborhoods).to(device)
                # Select only the needed rows from the feature and
                # indicator matrices
                if args.use_indicators:
                    x = torch.cat([features[batch_nodes],
                                   indicator_features[batch_nodes]],
                                  dim=1
                                  ).to(device)
                else:
                    x = features[batch_nodes].to(device)

                if gcn_gf.gcn_layers[1].out_channels == 1:
                    # Get probabilities for sampling each node
                    node_logits, _ = gcn_gf(x, local_neighborhoods)#[[local_neighborhoods[0], local_neighborhoods[1]],
                                                #[local_neighborhoods[0], local_neighborhoods[1]]],
                                            #[(len(batch_nodes), len(batch_nodes)),
                                            # (len(batch_nodes), len(batch_nodes))])
                    # Select logits for neighbor nodes only
                    node_logits = node_logits[node_map.map(neighbor_nodes)]

                    # Sample top k neighbors using the logits
                    b = Bernoulli(logits=node_logits.squeeze())
                    samples = torch.topk(b.probs, k=args.num_samples, dim=0, sorted=False)[1]
                    sample_mask = torch.zeros_like(node_logits.squeeze(), dtype=torch.float)
                    sample_mask[samples] = 1
                    sampled_neighboring_nodes = neighbor_nodes[sample_mask.bool().cpu()]

                    ## LADIES
                    # adj_row = adjacency[previous_nodes.squeeze(), :]
                    # pi = np.array(np.sum(adj_row.multiply(adj_row), axis=0))[0]
                    # p = pi / np.sum(pi)
                    # s_num = np.min([np.sum(p > 0), args.num_samples])
                    # sampled_neighboring_nodes = torch.tensor(np.random.choice(data.num_nodes, s_num, p=p,
                    #                                                           replace=False))
                else:
                    sampled_neighboring_nodes, _ = torch.sort(torch.tensor(
                        np.random.choice(neighbor_nodes, size=min(neighbor_nodes.size(0), args.num_samples),
                                         replace=False)))

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
                k_hop_edges_w_sloop = torch.cat([k_hop_edges, target_nodes.repeat(2, 1)], dim=1)
                sub_adj_size.append((len(previous_nodes), len(batch_nodes)))

                node_map.update(previous_nodes)
                local_hop_edges.append(node_map.map(k_hop_edges_w_sloop[0]))
                node_map.update(batch_nodes)
                local_hop_edges.append(node_map.map(k_hop_edges_w_sloop[1]))
                local_edge_indices.append(local_hop_edges)
                # Update the previous_nodes
                previous_nodes = batch_nodes.clone()

            all_nodes = node_map.values[all_nodes_mask]
            node_map.update(all_nodes)
            edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]

            x = features[all_nodes].to(device)
            # x = features[batch_nodes].to(device)
            logits_total, _ = gcn_c(x, edge_indices) #local_edge_indices, sub_adj_size)
            predictions = torch.argmax(logits_total, dim=1)
            predictions = predictions[node_map.map(target_nodes)]  # map back to original node IDs

            all_predictions = torch.cat([all_predictions, predictions], dim=0)

        all_predictions = all_predictions.cpu()
        targets = data.y[mask]

        accuracy = accuracy_score(targets, all_predictions)
        f1 = f1_score(targets, all_predictions, average='micro')

    return accuracy, f1