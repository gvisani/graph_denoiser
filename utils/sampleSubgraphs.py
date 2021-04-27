import torch
import torch_geometric
import random
TorchNodes = torch.LongTensor
TorchEdges = torch.LongTensor
TorchMappings = torch.LongTensor
TorchEdgeMask = torch.BoolTensor


def sampleSubgraphs(G: torch_geometric.data.Data, n: int) -> (int, list((TorchNodes, TorchEdges, TorchMappings, TorchEdgeMask))):
    subGraphs = []
    for i in range(n):
        node_idx = random.randint(0, G.num_nodes - 1)
        subGraphs.append(
            (node_idx, torch_geometric.utils.k_hop_subgraph(node_idx, 1, G.edge_index)))
    return subGraphs
