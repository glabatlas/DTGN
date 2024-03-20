import numpy as np
import pandas as pd
import networkx as nx
import os
from functools import partial
from pathos.pools import ProcessPool
import argparse


from pathway_analysis.TGMI import triple_interaction, p_adjust, discretize


# 将Pathway转换为pathway-[gene list]的形式
def read_pathway(dataset="LP", use_all=False, sep='\t'):
    path = f"./data/pathway/{dataset}/all.txt"
    pathways = pd.read_csv(path, sep=sep).values
    positive_terms = pd.read_csv(f"./data/pathway/{dataset}/positive_pathways.csv").iloc[:, 0].values.tolist()
    negative_terms = pd.read_csv(f"./data/pathway/{dataset}/negative_pathways.csv").iloc[:, 0].values.tolist()

    valid_terms = {term for term in positive_terms + negative_terms}
    pathways_dict = {}
    for line in pathways:
        if (line[0] not in valid_terms) and not use_all:
            continue
        if line[0] not in pathways_dict.keys():
            pathways_dict[line[0]] = [line[1]]
        else:
            pathways_dict[line[0]].append(line[1])

    return pathways_dict


# 获取特定阶段的tf-gene网络
def time_specific_network(path):
    edges = pd.read_csv(path).iloc[:, 0:2].values.tolist()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


# 获取特定阶段显著表达的tf
def time_specific_tf(path, p=0.05):
    tfs_p = pd.read_csv(path).values
    specific_tfs = [tfp[0].split('_')[0] for tfp in tfs_p if tfp[1] < p]
    return specific_tfs


def permutation_test(list1, list2, times=1000):
    observed_difference = np.mean(list1) - np.mean(list2)
    combined_data = np.concatenate((list1, list2)).reshape(-1)
    permuted_differences = []
    n = combined_data.shape[0] // 2
    for _ in range(times):
        np.random.shuffle(combined_data)
        permuted_group1 = combined_data[:n]
        permuted_group2 = combined_data[n:]
        permuted_difference = np.mean(permuted_group1) - np.mean(permuted_group2)
        permuted_differences.append(permuted_difference)

    # 计算P值
    p_value = (np.abs(permuted_differences) >= np.abs(observed_difference)).mean()
    return p_value


# 获取基因特定阶段的特征
def genes_feature(feat_path, mapping_path, i, stage=10):
    feat = pd.read_csv(feat_path)
    d = feat.shape[1] // stage
    # feat = np.array(feat.iloc[:, (i - 1) * d:i * d].values)
    feat = np.array(feat.values)
    mapping = np.array(pd.read_csv(mapping_path).iloc[:, 0:2].values)
    id_gene_dict = dict(mapping)
    gene_feat_dict = {}
    for i, specific_feat in enumerate(feat):
        gene_feat_dict[id_gene_dict[i]] = specific_feat
    return gene_feat_dict


def tf_pathway_score(dataset, run_id, permute_option="hidden_factor", stage=10, tf_p=0.05, threshold=0.05,
                     permutation=1000):
    pathway_tfs = read_pathway(dataset=dataset, use_all=True)
    pathway_name = [name for name in pathway_tfs.keys()]
    mapping_path = f"./data/{dataset}/mapping.csv"
    mapping = np.array(pd.read_csv(mapping_path).iloc[:, 0:2].values)
    gene_id_dict = dict(zip(mapping[:, 1], mapping[:, 0]))
    if (permute_option.split('_')[0] == 'exp'):
        using_exp = True
    else:
        using_exp = False
    if using_exp:
        feat = np.array(pd.read_csv(f'./out/{dataset}/features/exp/features.csv').values)
    else:
        feat = np.array(pd.read_csv(f'./out/{dataset}/features/{run_id}/features.csv').values)

    for i in range(1, stage):
        print(f"stage: {i}")
        specific_tfs = time_specific_tf(f"./out/{dataset}/permutation/{run_id}_{permute_option}/factor{i + 1}.csv",
                                        p=tf_p)


        if using_exp:
            specific_g = time_specific_network(f"../data/{dataset}/origin_net.csv")
            specific_feat = genes_feature(f'./out/{dataset}/features/exp/features.csv',
                                          f"./data/{dataset}/mapping.csv", i, stage=stage)
        else:
            specific_g = time_specific_network(f"./out/{dataset}/factor_link/{run_id}/factor{i + 1}.csv")
            # specific_g = time_specific_network(f"../data/{dataset}/origin_net.csv")
            specific_feat = genes_feature(f"./out/{dataset}/features/{run_id}/features.csv",
                                          f"./data/{dataset}/mapping.csv", i, stage=stage)

        for k in specific_feat.keys():
            specific_feat[k] = discretize(specific_feat[k])

        def per_pathway(pathway, tfs, func1, permutation):
            out = set()
            genes_set = set()
            path_genes = {t for t in pathway_tfs[pathway]}
            for tf in tfs:
                tf_genes = set(specific_g.neighbors(tf))
                in_genes = tf_genes.intersection(path_genes)
                others_genes = path_genes - in_genes
                tf_feat = specific_feat[tf]
                genes_set.update(in_genes)
                genes_set.update(others_genes)

                if len(in_genes) == 0:
                    continue
                for g2 in others_genes:
                    if g2 not in specific_feat.keys():
                        continue
                    for g1 in in_genes:
                        p_val = func1(tf_feat, specific_feat[g1], specific_feat[g2], permutation=permutation)
                        if p_val <= threshold:
                            out.update({tf, g1, g2})
            if len(genes_set) == 0:
                ret = [pathway, len(genes_set), 0, out]
            else:
                ret = [pathway, len(genes_set), len(out) / len(genes_set), out]
            return ret

        partial_wrapper = partial(per_pathway,
                                  tfs=specific_tfs,
                                  permutation=permutation,
                                  func1=triple_interaction)
        pool = ProcessPool(8)
        results = list(pool.amap(partial_wrapper, pathway_name).get())
        new_results = []
        for path_result in results:
            if path_result[2] == 0:
                continue
            genes = list(path_result[3])
            targets_id = [gene_id_dict[tg] for tg in genes]
            if using_exp:
                curr_exp = feat[targets_id][:, i]
                init_exp = feat[targets_id][:, 0]
            else:
                d = feat.shape[1] // stage
                curr_exp = feat[targets_id][:, i * d:(i + 1) * d]
                init_exp = feat[targets_id][:, 0:d]
            pvalues = permutation_test(curr_exp, init_exp)
            path_result[3] = pvalues
            new_results.append(path_result)

        pathway_tf_score = sorted(new_results, key=lambda x: x[3])
        dataframe = pd.DataFrame(pathway_tf_score, columns=["pathway", "nums", "percent", "pvalue"])
        out_path = f"../out/{dataset}/pathway_score/{run_id}/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        dataframe.to_csv(out_path + f"factor{i + 1}.csv", index=False)


def args_parsing():
    parser = argparse.ArgumentParser(description='Parsing args on main.py')
    parser.add_argument('-dataset', type=str, default="LP")
    parser.add_argument('-method', type=str, default='DyTGNets')
    parser.add_argument('-stage', type=int, default=10)
    parser.add_argument('-times', type=int, default=100)
    parser.add_argument('-threshold', type=float, default=0.01)
    return parser


if __name__ == '__main__':
    main_parser = args_parsing()
    args = main_parser.parse_args()

    dataset = args.dataset
    method = args.method
    stage = args.stage
    times = args.times
    threshold = args.threshold

    if method == "SSN":
        permute_option = "exp_factor"
    else:
        permute_option = "hidden_factor"

    tf_pathway_score(dataset, method, permute_option=permute_option, stage=stage, tf_p=threshold, permutation=times)
