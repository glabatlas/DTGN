# 预处理调控网络和基因表达网络

import pandas as pd
import numpy as np
import networkx as nx
import argparse


# 按均值和方差过滤表达数据
# mean=1 and var=0
def exp_filter(gene_exp, stage, mean=1, var=0):
    gene_exp = gene_exp.reshape((stage, -1)).mean(axis=-1)
    if np.mean(gene_exp) > mean and np.var(gene_exp) > var:
        return True
    else:
        return False


def exp_normalize(gene_exp, norm_type='log2'):
    gene_exp = gene_exp.astype(float)
    if norm_type == 'log2':
        return np.log2(gene_exp + 1)
    elif norm_type == 'id':
        return gene_exp
    elif norm_type == 'max-min':
        min_value = min(gene_exp)
        max_value = max(gene_exp)
        gene_exp = [(value - min_value) / (max_value - min_value) for value in gene_exp]
        return gene_exp
    elif norm_type == "LP-relog2":
        return np.log2(np.power(2, gene_exp) + 1)
    else:
        raise NotImplementedError


# 过滤数据集Mus_LP和Mus_LP2.
def gene_filter(dataset_name, stage, net_path, exp_path, map_path, norm_type='id', mean=1, var=0):
    # 取出网络中的边和节点
    edges = pd.read_csv(net_path).iloc[:, 0:2].values.tolist()
    edges_dir = pd.read_csv(net_path).iloc[:, 0:3].values.tolist()

    exp_data = pd.read_csv(exp_path, sep=',').values
    mapping_data = pd.read_csv(map_path).iloc[:, 0:2].values
    gene_ens_dict = dict(zip(mapping_data[:, 1], mapping_data[:, 0]))
    ens_exp_dict = dict(zip(exp_data[:, 0], exp_data[:, 1:]))

    temp_edges = edges_dir.copy()
    for u, v, d in temp_edges:
        if u not in gene_ens_dict.keys() \
                or v not in gene_ens_dict.keys() \
                or gene_ens_dict[u] not in ens_exp_dict.keys() \
                or gene_ens_dict[v] not in ens_exp_dict.keys() \
                or not exp_filter(ens_exp_dict[gene_ens_dict[u]], stage, mean, var) \
                or not exp_filter(ens_exp_dict[gene_ens_dict[v]], stage, mean, var) \
                or u == v:
            edges.remove([u, v])
            edges_dir.remove([u, v, d])

    G = nx.Graph()
    G.add_edges_from(edges)

    nodes = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    all_tfs = {e[0] for e in edges}
    nodes = {node for pair in edges for node in pair}
    out_gene_exp = []
    for g in nodes:
        if g in gene_ens_dict.keys():
            ens = gene_ens_dict[g]
            if ens in ens_exp_dict.keys() and exp_filter(ens_exp_dict[ens], stage):
                temp = [g, ens]
                norm_exp = exp_normalize(ens_exp_dict[ens], norm_type)
                temp.extend(norm_exp)
                out_gene_exp.append(temp)

    def sorted_func(x):
        is_gene = 1
        if x[0] in all_tfs:
            is_gene = 0
        return (is_gene, x[0])

    out_gene_exp = sorted(out_gene_exp, key=sorted_func)
    out_gene_exp = np.array(out_gene_exp)

    # 将节点映射到id，并保存映射
    gene_id_dict = dict(zip(out_gene_exp[:, 0], range(len(out_gene_exp))))
    id_edges = []
    for u, v in edges:
        id_edges.append([gene_id_dict[u], gene_id_dict[v]])

    id_name_ens = []
    for i, v in enumerate(out_gene_exp):
        id_name_ens.append([i, v[0], v[1]])

    origin_net = pd.DataFrame(edges_dir, columns=['source', 'target', 'Input'])
    df_edges = pd.DataFrame(id_edges, columns=['source', 'target'])
    df_nodes = pd.DataFrame(out_gene_exp)
    df_mapping = pd.DataFrame(id_name_ens)
    origin_net.to_csv(f"./data/{dataset_name}/origin_net.csv", index=False)
    df_edges.to_csv(f"./data/{dataset_name}/training_net.csv", index=False)
    df_nodes.to_csv(f"./data/{dataset_name}/training_node.csv", index=False)
    df_mapping.to_csv(f"./data/{dataset_name}/mapping.csv", index=False)


def args_parsing():
    parser = argparse.ArgumentParser(description='Parsing args on preprocessing.py')
    parser.add_argument('-dataset', type=str, default='LP')
    parser.add_argument('-stage', type=int, default=10)
    parser.add_argument('-norm_type', type=str, default='id')
    parser.add_argument('-mean', type=float, default=1)
    parser.add_argument('-var', type=float, default=0)
    return parser


if __name__ == '__main__':
    main_parser = args_parsing()
    args = main_parser.parse_args()

    dataset = args.dataset
    stage = args.stage
    norm_type = args.norm_type
    mean = args.mean
    var = args.var

    net_path = f"./data/{dataset}/raw_data/network.csv"
    exp_path = f"./data/{dataset}/raw_data/exp.csv"
    map_path = f"./data/{dataset}/raw_data/mapping.csv"
    gene_filter(dataset, stage, net_path, exp_path, map_path, norm_type=norm_type, mean=mean, var=var)
