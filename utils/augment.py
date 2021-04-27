import torch
import torch_geometric
import networkx as nx

TorchNodes = torch.LongTensor
TorchEdges = torch.LongTensor
TorchMappings = torch.LongTensor
TorchEdgeMask = torch.BoolTensor


def augment(s_1: TorchEdges, s_1_nodes: TorchNodes, s_2: TorchEdges,  s_2_nodes: TorchNodes, G_noisy: nx.Graph, t: list(("src", "tar"))) -> (TorchEdges, TorchEdges, list(("src", "tar"))):
    d1 = torch_geometric.data.Data(edge_index=s_1)
    d1.num_nodes = 0
    d2 = torch_geometric.data.Data(edge_index=s_2)
    d2.num_nodes = 0
    s_1_nx = torch_geometric.utils.to_networkx(d1, to_undirected=True)
    s_2_nx = torch_geometric.utils.to_networkx(d2, to_undirected=True)
    s_12_nx = G_noisy.subgraph(list(s_1_nx.nodes) + list(s_2_nx.nodes)).copy()

    y_s_12 = list(filter(lambda x: s_12_nx.has_edge(x[0], x[1]), t))

    s_1_nx.remove_edges_from(y_s_12)
    s_2_nx.remove_edges_from(y_s_12)

    return torch.LongTensor(list(s_1_nx.edges)).t().contiguous(), torch.LongTensor(list(s_2_nx.edges)).t().contiguous(), y_s_12
