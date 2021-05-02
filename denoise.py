import numpy as np
import networkx as nx
import torch
import wandb
import toolz
import random
from torch_geometric.nn.models.tgn import IdentityMessage, LastNeighborLoader
from torch_geometric import torch_geometric
from copy import deepcopy
from itertools import combinations
from utils import graphGenerators
from utils.sampleSubgraphs import sampleSubgraphs
from utils.augment import augment
from utils.addEdges import addEdges
from utils.removeEdges import removeEdges
from modules.relation import RelationModule
from modules.graphAttention import GraphAttentionEmbedding
from modules.linkPredictor import LinkPredictor
from modules.denoiserMemory import DenoiserMemory,  LastAggregator


wandb.init(project="denoiser")
G, G_noisy, C = graphGenerators.getRandomStartingNxGraphs(8, 30, 30)
torch_G = torch_geometric.utils.from_networkx(G_noisy)
torch_G_base = deepcopy(torch_G)
memory_dim = time_dim = embedding_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
memory = DenoiserMemory(
    torch_G.num_nodes*2,
    0,
    memory_dim,
    time_dim,
    message_module=IdentityMessage(0, memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=0,
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
rel_subgraph = RelationModule(100, 5)
rel_node = RelationModule(100, 5)

neighbor_loader = LastNeighborLoader(torch_G.num_nodes, size=25, device=device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(torch_G.num_nodes*2, dtype=torch.long, device=device)

wandb.watch(gnn, log_freq=100)
wandb.watch(memory)
wandb.watch(rel_node)


def train(num_rounds, num_edges_to_add, lam, tau, num_nodes, confident_size):
    memory.train()
    gnn.train()
    link_pred.train()

    torch_G = deepcopy(torch_G_base)
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    neighbor_loader.insert(torch_G.edge_index[0], torch_G.edge_index[1])
    log_metrics = log_metrics_with_G(G)

    T = deepcopy(C)
    T_random = deepcopy(C)
    total_loss = 0
    loss_trace = []
    T_trace = []
    for i in range(num_rounds):
        print("Length of T", len(T))
        torch_G = deepcopy(torch_G_base)
        T = deepcopy(C)
        T_random = deepcopy(C)
        subgraphs = sampleSubgraphs(torch_G, 30)
        for S1, S2 in combinations(subgraphs, 2):
            optimizer.zero_grad()
            # region learning_by_augmentation.
            S_1_aug, S_2_aug, y, y_rem, y_neg = augment(
                S1[1][1], S1[1][0], S2[1][1], S2[1][0], G_noisy, T)

            n_id = torch.cat(
                [S_1_aug[0], S_2_aug[0], S1[1][0], S2[1][0]]).unique()
            # n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id, y_rem)
            # print(last_update)
            try:
                z = gnn(z, last_update, torch.cat(
                    [assoc[S_1_aug], assoc[S_2_aug]], dim=1), i)
            except:
                print('In gnn')
                print([assoc[S_1_aug], assoc[S_2_aug]])
                print(S1[1][1])
                print(S2[1][1])
                print(S_1_aug)
                print(S_2_aug)
                exit(1)
            try:
                memory.update_state(torch.cat([S_1_aug[0], S_2_aug[0]]), torch.cat(
                    [S_1_aug[1], S_2_aug[1]]), torch.empty(S_1_aug.size(1) + S_2_aug.size(1)).fill_(i).type(torch.int64), torch.empty(S_1_aug.size(1) + S_2_aug.size(1), 0), y_rem)
            except:
                print('In memory update')
                print(S1[1][1])
                print(S2[1][1])
                print(S_1_aug)
                print(S_2_aug)
                exit(1)
            # assoc[n_id] = torch.arange(n_id.size(0), device=device)
            v_1 = torch.sum(z[assoc[S_1_aug[0]]], 0) / S_1_aug[0].size(0)
            v_2 = torch.sum(z[assoc[S_2_aug[0]]], 0) / S_2_aug[0].size(0)
            mu_prime_G = rel_subgraph(v_1, v_2)
            # N_1, N_2 = assoc[n_id], assoc[n_id]
            # assoc_s1[S1[1][0]] = torch.arange(S1[1][0].size(0), device=device)
            # assoc_s2[S2[1][0]] = torch.arange(S2[1][0].size(0), device=device)

            mu_p = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            truth = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            for n1, n2 in y:
                truth[assoc[n1], assoc[n2]] = 1.0

            n_id_combos_src = torch.combinations(
                n_id, r=2, with_replacement=False)
            n_id_combos_dst = torch.flip(n_id_combos_src, [1])
            n_id_combos = torch.cat(
                (n_id_combos_src, n_id_combos_dst), dim=0)
            n_id_A = n_id_combos[:, 0]
            n_id_B = n_id_combos[:, 1]
            z_A = z[assoc[n_id_A]]
            z_B = z[assoc[n_id_B]]

            n_id_1 = S_1_aug[1][0]
            n_id_2 = S_2_aug[1][0]
            v_A = torch.empty(z_A.shape)
            v_B = torch.empty(z_B.shape)

            mask_A1 = mask = torch.BoolTensor(
                [True if n in n_id_1 else False for n in n_id_A])
            mask_A2 = mask = torch.BoolTensor(
                [True if n in n_id_2 else False for n in n_id_A])
            mask_B1 = mask = torch.BoolTensor(
                [True if n in n_id_1 else False for n in n_id_B])
            mask_B2 = mask = torch.BoolTensor(
                [True if n in n_id_2 else False for n in n_id_B])

            v_A[mask_A1] = v_1
            v_A[mask_A2] = v_2
            v_B[mask_B1] = v_1
            v_B[mask_B2] = v_2

            mu_p_temp = rel_node(mu_prime_G*(v_A + z_A),
                                 mu_prime_G*(v_B + z_B))

            # print(mu_p_temp.squeeze(1))

            loss = None
            num_predictions = 0
            for idx, edge in enumerate(n_id_combos):
                if tuple(edge) in y:
                    if loss is None:
                        loss = tau*(1.0 - mu_p_temp[idx][0])**2
                    else:
                        loss += tau*(1.0 - mu_p_temp[idx][0])**2
                    num_predictions += 1
                elif edge in torch_G.edge_index.t() and tuple(edge) not in y:
                    if loss is None:
                        loss = lam*(0.5 - mu_p_temp[idx][0])**2
                    else:
                        loss += lam*(0.5 - mu_p_temp[idx][0])**2
                    if lam != 0.0:
                        num_predictions += 1
                elif tuple(edge) in y_neg:
                    if loss is None:
                        loss = (0.0 - mu_p_temp[idx][0])**2
                    else:
                        loss += (0.0 - mu_p_temp[idx][0])**2
                    num_predictions += 1

            if loss is None:
                continue

            loss /= num_predictions
            loss.backward()
            optimizer.step()
            memory.detach()
            # endregion

            n_id = torch.cat([S1[1][0], S2[1][0]]).unique()
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id, [])
            # print(last_update)
            z = gnn(z, last_update, torch.cat(
                [assoc[S1[1][1]], assoc[S2[1][1]]], dim=1), i)
            # print('updating memory')
            # memory.update_state(torch.cat([S1[1][1][0], S2[1][1][0]]), torch.cat(
            # [S1[1][1][1], S2[1][1][1]]), torch.empty(S1.size(1) + S2.size(1)).fill_(i).type(torch.int64), torch.empty(S1.size(1) + S2.size(1), 0), [])

            v_1 = torch.sum(z[assoc[S1[1][0]]], 0) / S1[1][0].size(0)
            v_2 = torch.sum(z[assoc[S2[1][0]]], 0) / S2[1][0].size(0)
            mu_prime_G = rel_subgraph(v_1, v_2)
            # N_1, N_2 = assoc[n_id], assoc[n_id]
            # assoc_s1[S1[1][0]] = torch.arange(S1[1][0].size(0), device=device)
            # assoc_s2[S2[1][0]] = torch.arange(S2[1][0].size(0), device=device)

            mu_p = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            truth = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            for n1, n2 in y:
                truth[assoc[n1], assoc[n2]] = 1.0

            n_id_combos_src = torch.combinations(
                n_id, r=2, with_replacement=False)
            n_id_combos_dst = torch.flip(n_id_combos_src, [1])
            n_id_combos = torch.cat(
                (n_id_combos_src, n_id_combos_dst), dim=0)
            n_id_A = n_id_combos[:, 0]
            n_id_B = n_id_combos[:, 1]
            z_A = z[assoc[n_id_A]]
            z_B = z[assoc[n_id_B]]

            n_id_1 = S1[1][0]
            n_id_2 = S2[1][0]
            v_A = torch.empty(z_A.shape)
            v_B = torch.empty(z_B.shape)

            mask_A1 = mask = torch.BoolTensor(
                [True if n in n_id_1 else False for n in n_id_A])
            mask_A2 = mask = torch.BoolTensor(
                [True if n in n_id_2 else False for n in n_id_A])
            mask_B1 = mask = torch.BoolTensor(
                [True if n in n_id_1 else False for n in n_id_B])
            mask_B2 = mask = torch.BoolTensor(
                [True if n in n_id_2 else False for n in n_id_B])

            v_A[mask_A1] = v_1
            v_A[mask_A2] = v_2
            v_B[mask_B1] = v_1
            v_B[mask_B2] = v_2

            mu_p_temp = rel_node(mu_prime_G*(v_A + z_A),
                                 mu_prime_G*(v_B + z_B))
#
            # print(mu_p_temp.squeeze(1))
            # print(loss)

            # n_id_combos: all edge combinations
            # mu_p_temp: scores for each edge

            # add to set of edges
            top_edge_values, top_edge_indices = torch.topk(
                mu_p_temp.squeeze(1), min(num_edges_to_add, mu_p_temp.squeeze(1).size(0)))
            addEdges(torch_G, n_id_combos[top_edge_indices])
            for edge in n_id_combos[top_edg

                                    # add random edges
                                    while len(T_random) < len(T):
                                    e1= random.randint(0, torch_G.num_nodes-1)
                                    e2= random.randint(0, torch_G.num_nodes-1)
                                    while e1 == e2:
                                    e2= random.randint(0, torch_G.num_nodes-1)
                                    T_random.add((e1, e2))
                                    e_indices]:
                T.add(tuple(edge.tolist()))

            # remove from set of confident edges (without removing edges in C)
            bottom_edge_values, bottom_edge_indices = torch.topk(
                mu_p_temp.squeeze(1), min(num_edges_to_add, mu_p_temp.squeeze(1).size(0)), largest=False)
            edges_to_remove = []
            for edge in n_id_combos[bottom_edge_indices]:
                if edge not in C and edge in T:
                    T.remove(tuple(edge.tolist()))
                    edges_to_remove.append(edge)

            # remove random edges
            while len(T_random) > len(T):
                edge = random.choice(tuple(T_random))
                while edge in C:
                    edge = random.choice(tuple(T_random))
                T_random.remove(edge)

            wandb.log({'loss': loss.item(), 'T': len(T), 'round': i,
                      'to_add': num_edges_to_add, 'lam': lam, 'tau': tau, 'num_nodes': num_nodes, 'confident_size': confident_size
                       })

            log_metrics(T)(T_random)
            loss_trace.append(loss.item())
            T_trace.append(len(T))

    return loss_trace, T_trace


SRC = str
TAR = str


@toolz.curry
def log_metrics_with_G(G: nx.Graph, T: set((SRC, TAR)), T_random: set((SRC, TAR))) -> None:
    edges = G.edges
    edges_set = set(edges)
    intersection = edges_set & T
    precision = len(intersection) / len(T)
    recall = len(intersection) / len(edges_set)

    intersection_random = edges_set & T_random
    precision_random = len(intersection_random) / len(T_random)
    recall_random = len(intersection_random) / len(edges_set)
    wandb.log({'precision': precision, 'recall': recall,
              'precision_random': precision_random, 'recall_random': recall_random})


if __name__ == "__main__":
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    print(len(G.edges))
    # test = torch.LongTensor([[1, 2], [2, 4], [3, 6]])
    # print(torch_G.edge_index.shape)
    # addEdges(torch_G, test)

    # number of rounds -- number of edges to add/remove in T -- loss contrib. of non-confident edges -- loss contrib. of positive edges
    # print(torch_G.edge_index.s5ape)0

    loss_trace, T_trace = train(5, 1, 0.25, 0.5, len(G.nodes), 30)
