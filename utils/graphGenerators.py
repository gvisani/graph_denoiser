import networkx as nx
import random
import toolz
from typing import Union

OpType = Union["add", "remove"]


def getRandomStartingNxGraphs(n: int, c: int, noisy_e_count: int) -> (nx.Graph, nx.Graph, list(("u", "v"))):
    '''
    Parameters
    ----------
    n: int 
       the number of nodes. n >= 5
    c: int
       the size of the confident set. 
       Note: This is a best of effort confident set size and will be
       <= c.
    noisy_e_count: int
       number of noisy edges in the network
    '''
    # Geometric Network, Relaxed Cavemen. Check networkx for community based graphs
    G = nx.generators.community.relaxed_caveman_graph(n, 12, 0.2)
    G_noisy = G.copy()
    edges = list(G.edges)
    assert (len(edges) > 0)
    nonedges = list(nx.non_edges(G))
    addOrRemoveEdge = addOrRemoveRandomEdgeWithAllEdges(edges)(nonedges)
    edge_operations = list(
        map(lambda x: addOrRemoveEdge(G), range(noisy_e_count)))

    for op, edge in edge_operations:
        mutateGraphWithEdge(G_noisy, op, edge)

    true_edges = set(G_noisy.edges) & set(edges)
    c_true_confident_edges = random.sample(true_edges, c)
    for node in G_noisy.nodes():
        G_noisy.nodes[node]['degree'] = nx.degree(G, node)
    return (G, G_noisy, set(c_true_confident_edges))


@toolz.curry
def addOrRemoveRandomEdgeWithAllEdges(edges: nx.edges, nonedges: nx.edges, G: nx.Graph) -> (OpType, ("src", "dst")):
    operation = random.choice(["remove"])
    chosen_edge = random.choice(edges)
    if operation == "remove":
        return ("remove", chosen_edge)
    if operation == "add":
        chosen_nonedge = random.choice(
            [x for x in nonedges if chosen_edge[0] == x[0]])
        return ("add", chosen_nonedge)


def mutateGraphWithEdge(G: nx.Graph, op: OpType, edge: ("src", "dst")) -> None:
    if op == "add":
        G.add_edge(edge[0], edge[1])
    elif op == "remove":
        try:
            G.remove_edge(edge[0], edge[1])
        except:
            pass
