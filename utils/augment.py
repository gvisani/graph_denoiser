import torch
import torch_geometric
import networkx as nx
import random

TorchNodes = torch.LongTensor
TorchEdges = torch.LongTensor
TorchMappings = torch.LongTensor
TorchEdgeMask = torch.BoolTensor


def augment(s_1: TorchEdges, s_1_nodes: TorchNodes, s_2: TorchEdges,  s_2_nodes: TorchNodes, G_noisy: nx.Graph, t: set(("src", "tar"))) -> (TorchEdges, TorchEdges, list(("src", "tar")), list(("src", "tar"))):
    d1 = torch_geometric.data.Data(edge_index=s_1)
    d1.num_nodes = 0
    d2 = torch_geometric.data.Data(edge_index=s_2)
    d2.num_nodes = 0
    s_1_nx = torch_geometric.utils.to_networkx(d1, to_undirected=False)
    s_2_nx = torch_geometric.utils.to_networkx(d2, to_undirected=False)
    s_12_nx = G_noisy.subgraph(list(s_1_nx.nodes) + list(s_2_nx.nodes)).copy()

    y_s_12 = list(filter(lambda x: s_12_nx.has_edge(x[0], x[1]), t))

    y_s_12_sample = []
    if len(y_s_12) != 0:
        y_s_12_sample = random.sample(y_s_12, random.randint(
            0,  max(1, min(len(y_s_12) - 1, min((s_1[0].size(0) // 2) - 1, (s_2[0].size(0) // 2) - 1)))))

    if (len(s_1_nx.edges)) > 2:
        s_1_nx.remove_edges_from(y_s_12_sample)
        s_1_nx.remove_edges_from(list(map(swap, y_s_12_sample)))
    if (len(s_2_nx.edges)) > 2:
        s_2_nx.remove_edges_from(y_s_12_sample)
        s_2_nx.remove_edges_from(list(map(swap, y_s_12_sample)))

    y_neg = [(random.choice(list(s_1_nx.nodes) + list(s_2_nx.nodes)), random.randint(
        len(G_noisy.nodes) + 1, len(G_noisy.nodes) * 2) - 1) for _ in range(len(y_s_12_sample))]

    y_neg_s_1 = list(filter(lambda x: x[0] in s_1_nx.nodes, y_neg))
    y_neg_s_2 = list(filter(lambda x: x[0] in s_2_nx.nodes, y_neg))
    s_1_nx.add_edges_from(y_neg_s_1)
    s_1_nx.add_edges_from(list(map(swap, y_neg_s_1)))
    s_2_nx.add_edges_from(y_neg_s_2)
    s_2_nx.add_edges_from(list(map(swap, y_neg_s_2)))

    return torch.LongTensor(list(s_1_nx.edges)).t().contiguous(), torch.LongTensor(list(s_2_nx.edges)).t().contiguous(), y_s_12, y_s_12_sample, y_neg


def swap(x):
    return (x[1], x[0])
