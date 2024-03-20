import dgl
import networkx as nx

def create_network(link_pairs, num_node, is_bidirected=True, add_loop=True, ret_adj=False):
    g = dgl.graph(link_pairs, num_nodes=num_node)
    g = dgl.remove_self_loop(g)
    if is_bidirected:
        g = dgl.to_bidirected(g)
    if add_loop:
        g = dgl.add_self_loop(g)
    if not ret_adj:
        return g
    else:
        return g.adj().to_dense()


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