import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from ..utils.file_operation import save_df

def delta_pcc(features, t, stage):
    """
    Calculating the delta pcc between t stages and (t-1) stages
    """
    global p_value
    d = features.shape[1] // stage
    assert 1 <= t <= stage
    five_features = features[:, [i for i in range(stage * d) if (t * d <= i or i < (t - 1) * d)]]
    pcc_n = np.corrcoef(five_features)
    k2 = 0
    N = d * (stage - 1)

    total_delta = []
    total_pcc = []
    z_score_list = []
    for i in range(d):
        perturbed_feat = np.concatenate((five_features, features[:, d * (t - 1) + i].reshape((-1, 1))), axis=1)
        pcc_perturbed = np.corrcoef(perturbed_feat)
        delta = pcc_perturbed - pcc_n
        total_delta.append(delta)
        total_pcc.append(pcc_n)
        eps = 1e-20
        z_score = delta / ((1 - pcc_n ** 2 + eps) / (N - 1))
        z_score_list.append(z_score)
        if d == 1:
            # Using Z-test when d == 1
            p_value = stats.norm.sf(abs(z_score)) * 2
        else:
            # Using Chi-square test when d > 1
            k2 += z_score ** 2

    # The degree of freedom is d - 1
    if d > 1:
        p_value = stats.chi2.sf(k2, d - 1)
    return p_value, total_delta, total_pcc


class GenePair:
    """
    The edges class with its significant value.
    """

    def __init__(self, tf, gene, value):
        self.tf = tf
        self.gene = gene
        self.value = value

    def __lt__(self, other):
        if self.value < other.value:
            return True
        else:
            return False

    def __str__(self):
        return self.gene + '\t' + self.tf + '\t' + str(self.value)


def get_stage_edges(t, feats, idx2sybol, edges, stage, threshold=0.01):
    pvalues, deltas, pcc_n = delta_pcc(feats, t, stage)
    factor_links = []

    for u, v in edges:
        p = pvalues[u, v]
        x = idx2sybol[u]
        y = idx2sybol[v]
        pair = GenePair(x, y, p)
        factor_links.append(pair)

    factor_links = sorted(factor_links, reverse=False, key=lambda x: x.value)
    pvalue_list = [f.value for f in factor_links]
    corr_pvalue = pvalue_list
    corr_out = []
    for i, f in enumerate(factor_links):
        if corr_pvalue[i] <= threshold:
            f.value = corr_pvalue[i]
            corr_out.append(f)
    return corr_out, deltas, pcc_n


def construct_network(edges):
    nodes = {node for pair in edges for node in pair}
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    C = sorted(nx.connected_components(G), key=len, reverse=True)
    print(G, end=', ')
    if len(C) == 0:
        C = [[]]
    # print(f'Maximum connected subgraph node count/total node count：{len(C[0])}/{len(nodes)},number of subgraphs:: {len(C)}')


def get_factor_grn(name, feats, edges, idx2sybol, stage, threshold):
    """
    Constructs the dynamic TF-Gene network for each stage.

    Parameters:
    name (str): The name of the dataset or experiment.
    feats (Tensor): The features for each node.
    edges (Tensor): The edge indices.
    idx2sybol (dict): A mapping from index to symbol.
    stage (int): The number of stages to process.
    threshold (float): The threshold for filtering edges.

    Returns:
    All outputs are saved in the "./out/{name}/dynamicTGNs" directory.

    Example:
    >>> get_factor_grn("experiment", feats, edges, idx2sybol, 5, 0.05)
    """
    total_delta = []
    total_pcc = []
    for i in range(1, stage + 1):
        factor_link, deltas, pcc_n = get_stage_edges(i, feats, idx2sybol, edges, stage, threshold)
        total_delta += deltas
        total_pcc += pcc_n
        data = []
        out_edges = []
        for link in factor_link:
            data.append([link.tf, link.gene, link.value])
            out_edges.append([link.tf, link.gene])
        construct_network(out_edges)
        dataframe = pd.DataFrame(data)
        save_df(dataframe, f"./out/{name}/dynamicTGNs", f"factor{i}.csv", header=['Source', 'Target', 'PValue'])
