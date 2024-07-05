import dgl
import networkx as nx


def is_conneted_graph(edges):
    nodes = {node for pair in edges for node in pair}
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    C = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(C) == 1:
        return True, len(C)
    else:
        return False, len(C[0])