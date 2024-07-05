# @Author : CyIce
# @Time : 2024/6/24 14:55

import numpy as np
import pandas as pd
import networkx as nx


def exp_normalize(exp, norm_type='id'):
    """
    Transform the expression to logarithmic space.
    """
    exp = np.array(exp, dtype=float)
    if norm_type == 'id':
        return exp
    elif norm_type == 'log2':
        return np.log2(exp + 1)
    elif norm_type == 'log10':
        return np.log10(exp + 1)
    else:
        raise NotImplementedError


def gene_filter(exp, mean, var):
    """
    Input a list of gene expressions, with each row containing the gene name followed by the gene expression values at
    various time points.
    """
    filed_exp = []
    for gene_exp in exp:
        if np.mean(gene_exp[1:]) >= mean and np.var(gene_exp[1:]) >= var:
            filed_exp.append(gene_exp)
    return filed_exp


def preprocessing(exp_path, net_path, mean, var, norm_type='id'):
    """
    Filter gene expression based on mean and variance, ensuring the connectivity of the network.
    """
    # Read the network and gene expression data
    edges = pd.read_csv(net_path).iloc[:, 0:2].values.tolist()
    all_tfs = {e[0] for e in edges}
    exp_data = pd.read_csv(exp_path).values
    exp_data = gene_filter(exp_data, mean, var)
    s2e = {row[0]: row[1:] for row in exp_data}

    edges = [edge for edge in edges if edge[0] in s2e.keys() and edge[1] in s2e.keys()]
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # Obtain the maximum connected component
    nodes = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    graph = graph.subgraph(nodes)
    out_exp = []
    for node in graph.nodes:
        gene_exp = [node, exp_normalize(s2e[node], norm_type)]
        out_exp.append(gene_exp)

    # Sort the gene expression data by gene name, and then sort the TFs and non-TFs separately
    def sorted_func(x):
        is_gene = 1
        if x[0] in all_tfs:
            is_gene = 0
        return is_gene, x[0]

    out_exp = sorted(out_exp, key=sorted_func)
    return out_exp, graph.edges


# if __name__ == '__main__':
#     exp, edges = preprocessing(1)
