import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from utils.file_operation import save_df

from utils.draw_pcc_hist import draw_hist


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

    for i in range(d):
        perturbed_feat = np.concatenate((five_features, features[:, d * (t - 1) + i].reshape((-1, 1))), axis=1)
        pcc_perturbed = np.corrcoef(perturbed_feat)
        delta = pcc_perturbed - pcc_n

        # drawing the histogram of delta pcc.
        draw_hist(delta, title=f"iPSC-{t}", bins=100,saved_path="./out/img")
        eps = 1e-20
        z_score = delta / ((1 - pcc_n ** 2 + eps) / (N - 1))
        if d == 1:
            # Using Z-test when d == 1
            p_value = stats.norm.sf(abs(z_score)) * 2
        else:
            # Using Chi-square test when d > 1
            k2 += z_score ** 2
    if d > 1:
        p_value = stats.chi2.sf(k2, d - 1)
    return p_value


class GenePair:
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
    pvalues = delta_pcc(feats, t, stage)
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
    return corr_out


def construct_network(edges):
    nodes = {node for pair in edges for node in pair}
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    C = sorted(nx.connected_components(G), key=len, reverse=True)
    print(G, end=', ')
    if len(C) == 0:
        C = [[]]
    print(f'最大连通子图节点数/节点总数：{len(C[0])}/{len(nodes)},子图数量: {len(C)}')


def get_factor_grn(name, feats, edges, idx2sybol, stage, threshold):
    """
    Constructing the dynamic TF-Gene network for each stage.
    """
    # feats = np.array(pd.read_csv(f'out/{name}/features.csv').values)
    for i in range(1, stage + 1):
        factor_link = get_stage_edges(i, feats, idx2sybol, edges, stage, threshold)
        data = []
        out_edges = []
        for link in factor_link:
            data.append([link.tf, link.gene, link.value])
            out_edges.append([link.tf, link.gene])
        construct_network(out_edges)
        dataframe = pd.DataFrame(data)
        save_df(dataframe, f"./out/{name}/dynamicTGNs", f"factor{i}.csv", header=['Source', 'Target', 'PValue'])

    exit(0)