import numpy as np
import torch
import wandb
from torch_geometric.nn.models.tgn import IdentityMessage, LastNeighborLoader
from torch_geometric import torch_geometric
from itertools import combinations
from utils import graphGenerators
from utils.sampleSubgraphs import sampleSubgraphs
from utils.augment import augment
from modules.relation import RelationModule
from modules.graphAttention import GraphAttentionEmbedding
from modules.linkPredictor import LinkPredictor
from modules.denoiserMemory import DenoiserMemory,  LastAggregator

wandb.init(project="denoiser")
G, G_noisy, C = graphGenerators.getRandomStartingNxGraphs(150, 30, 30)
torch_G = torch_geometric.utils.from_networkx(G_noisy)
memory_dim = time_dim = embedding_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
memory = DenoiserMemory(
    torch_G.num_nodes,
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
assoc = torch.empty(torch_G.num_nodes, dtype=torch.long, device=device)

wandb.watch(gnn, log_freq=100)
wandb.watch(memory)
wandb.watch(rel_node)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    neighbor_loader.insert(torch_G.edge_index[0], torch_G.edge_index[1])

    total_loss = 0
    for i in range(5):
        subgraphs = sampleSubgraphs(torch_G, 30)
        for S1, S2 in combinations(subgraphs, 2):
            print('round start')
            optimizer.zero_grad()
            S_1_aug, S_2_aug, y = augment(
                S1[1][1], S1[1][0], S2[1][1], S2[1][0], G_noisy, C)
            n_id = torch.cat([S1[1][0], S2[1][0]]).unique()
            # n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id, y)
            print(last_update)
            z = gnn(z, last_update, torch.cat(
                [assoc[S_1_aug], assoc[S_2_aug]], dim=1), i)
            print('updating memory')
            memory.update_state(torch.cat([S_1_aug[0], S_2_aug[0]]), torch.cat(
                [S_1_aug[1], S_2_aug[1]]), torch.empty(S_1_aug.size(1) + S_2_aug.size(1)).fill_(i).type(torch.int64), torch.empty(S_1_aug.size(1) + S_2_aug.size(1), 0), y)

            # assoc[n_id] = torch.arange(n_id.size(0), device=device)
            v_1 = torch.sum(z[assoc[S1[1][0]]], 0) / S1[1][0].size(0)
            v_2 = torch.sum(z[assoc[S2[1][0]]], 0) / S2[1][0].size(0)
            mu_prime_G = rel_subgraph(v_1, v_2)
            N_1, N_2 = assoc[n_id], assoc[n_id]
            # assoc_s1[S1[1][0]] = torch.arange(S1[1][0].size(0), device=device)
            # assoc_s2[S2[1][0]] = torch.arange(S2[1][0].size(0), device=device)

            mu_p = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            truth = torch.empty(n_id.size(0), n_id.size(0)).fill_(np.nan)
            for n1, n2 in y:
                truth[assoc[n1], assoc[n2]] = 1.0

            def nanmean(v, *args, inplace=False, **kwargs):
                if not inplace:
                    v = v.clone()
                is_nan = torch.isnan(v)
                v[is_nan] = 0
                return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

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

            loss = None
            num_pos = 0
            for idx, edge in enumerate(n_id_combos):
                if tuple(edge) in y:
                    if loss is None:
                        loss = (1.0 - mu_p_temp[idx][0])**2
                    else:
                        loss += (1.0 - mu_p_temp[idx][0])**2
                    num_pos += 1

            if loss is None:
                continue

            print("backpropping")
            loss /= num_pos

            # loss = None
            # for n1 in N_1:
            #     for n2 in N_2:
            #         # Check if is nan and compute loss depending n that.
            #         if loss is None:
            #             loss = (truth[n1, n2] - rel_node(
            #                 mu_prime_G*(v_1 + z[n1]), mu_prime_G*(v_2 + z[n2])))**2
            #         else:
            #             loss = torch.nansum(torch.Tensor([loss, (truth[n1, n2] - rel_node(
            #                 mu_prime_G*(v_1 + z[n1]), mu_prime_G*(v_2 + z[n2])))**2]))

            # # loss = nanmean(torch.flatten((truth - mu_p)**2), inplace=True)
            loss.backward()
            optimizer.step()
            memory.detach()
            print(loss)

            # for n1, n2 in y:
            #     truth[assoc_s1[n1], assoc_s2[n2]] = 1.0


if __name__ == "__main__":
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    train()
