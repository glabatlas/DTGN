# @Author : CyIce
# @Time : 2024/6/25 17:53


import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from history_code.utils_bak import file_operation


# 计算某个时期特征向量和其他时期的差异
def delta_pcc(features, t, stage=6, ):
    global p_value, reg_model
    d = features.shape[1] // stage
    # 阶段从1开始,t结束
    assert 1 <= t <= stage
    five_features = features[:, [i for i in range(stage * d) if (t * d <= i or i < (t - 1) * d)]]
    pcc_n = np.corrcoef(five_features)
    k2 = 0
    N = d * (stage - 1)
    # d>1待修改
    for i in range(d):
        perturbed_feat = np.concatenate((five_features, features[:, d * (t - 1) + i].reshape((-1, 1))),
                                        axis=1)
        pcc_perturbed = np.corrcoef(perturbed_feat)
        delta = pcc_perturbed - pcc_n
        eps = 1e-20
        z_score = delta / ((1 - pcc_n ** 2 + eps) / (N - 1))
        # 如果每个时间段特征维度为1维，使用z检验
        if d == 1:
            p_value = stats.norm.sf(abs(z_score)) * 2  # 双侧检验
        else:
            k2 += z_score ** 2
            p_value = stats.chi2.sf(k2, d - 1)
        reg_model = np.where(delta > 0, 1, -1)

    return p_value, reg_model


class GenePair:
    def __init__(self, tf, gene, value, model):
        self.tf = tf
        self.gene = gene
        self.value = value
        self.model = model

    def __lt__(self, other):
        if self.value < other.value:
            return True
        else:
            return False

    def __str__(self):
        return self.gene + '\t' + self.tf + '\t' + str(self.value)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def get_stage_genepairs(t, all_features, stage=6, threshold=0.01, mapping_path='', edges_path='',
                        using_net=True):
    mapping = np.array(pd.read_csv(mapping_path).iloc[:, 0:2].values)
    edges = pd.read_csv(edges_path).values
    id_gene_dict = dict(mapping)
    pvalues, reg_model = delta_pcc(all_features, t, stage)
    factor_links = []

    if using_net:
        for u, v in edges:
            model = reg_model[u, v]
            p = pvalues[u, v]
            x = id_gene_dict[u]
            y = id_gene_dict[v]
            pair = GenePair(x, y, p, model)
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


def get_factor_grn(dataset, stage=6, run_id=1, threshold=0.01, mapping_path='', edges_path='', using_feat=True):
    feat_name = "features"
    if not using_feat:
        feat_name = "de_feat"
    feat = np.array(pd.read_csv(f'out/{dataset}/features/{run_id}/{feat_name}.csv').values)
    for i in range(1, stage + 1):
        factor_link = get_stage_genepairs(t=i, all_features=feat, stage=stage, threshold=threshold,
                                          mapping_path=mapping_path,
                                          edges_path=edges_path)
        data = []
        edges = []
        for f in factor_link:
            data.append([f.tf, f.gene, f.value, f.model])
            edges.append([f.tf, f.gene])
        construct_network(edges)
        file_operation.save_features(path=f"out/{dataset}/factor_link/{run_id}",
                                     filename=f'factor{i}.csv', feat=data,
                                     columns=['source', 'target', 'p-value', 'model'])
