import torch
import torch_geometric
import networkx as nx

TorchNodes = torch.LongTensor
TorchEdges = torch.LongTensor
TorchMappings = torch.LongTensor
TorchEdgeMask = torch.BoolTensor


def addEdges(G: torch_geometric.data.Data, edges: torch.LongTensor):

    dense = torch_geometric.utils.to_dense_adj(G.edge_index)[0]

    # Do addition to dense matrix
    for edge in edges:
        dense[edge[0], edge[1]] = 1

    G.edge_index = torch_geometric.utils.dense_to_sparse(dense)[0]

    return
