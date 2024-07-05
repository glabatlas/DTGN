import numpy as np
import pandas as pd
import networkx as nx
import os
from functools import partial
from pathos.pools import ProcessPool
import argparse

from utils.TGMI import triple_interaction, discretize
from utils.file_operation import save_df


# Reading pathway annotation.
def read_pathway(annotation_path, positive_pathway, negative_pathway, use_all=False, sep='\t'):
    pathways = pd.read_csv(annotation_path, sep=sep).values
    positive_terms = pd.read_csv(positive_pathway).iloc[:, 0].values.tolist()
    negative_terms = pd.read_csv(negative_pathway).iloc[:, 0].values.tolist()

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


# Constructing a time-specific network.
def time_specific_network(path):
    edges = pd.read_csv(path).iloc[:, 0:2].values.tolist()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


# Identify the key transcription factors at each developmental stage.
def time_specific_tf(path, p=0.05):
    tfs_p = pd.read_csv(path).values
    specific_tfs = [tfp[0].split('_')[0] for tfp in tfs_p if tfp[1] < p]
    return specific_tfs


# Using a permutation test to determine the significance of two feature lists.
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

    # Calculating the p value.
    p_value = (np.abs(permuted_differences) >= np.abs(observed_difference)).mean()
    return p_value


# Calculate the significance score for each pathway.
def tf_pathway_score(name, annotation_path, positive_pathway_path, negative_pathway_path, stage=10,
                     tf_p=0.05, threshold=0.05, permutation=1000):
    pathway_tfs = read_pathway(annotation_path, positive_pathway_path, negative_pathway_path, use_all=True)
    pathway_name = [name for name in pathway_tfs.keys()]

    feat = pd.read_csv(f'../out/{name}/features.csv')
    feat_dict = dict(zip(feat.iloc[:, 0], feat.iloc[:, 1:].values.tolist()))

    for k, v in feat_dict.items():
        feat_dict[k] = discretize(v)

    for i in range(1, stage):
        print(f"stage: {i}")
        specific_tfs = time_specific_tf(f"../out/{name}/permutation/stage{i + 1}.csv", tf_p)
        specific_g = time_specific_network(f"../out/{name}/dynamicTGNs/factor{i + 1}.csv")

        def per_pathway(pathway, tfs, func1, permutation):
            out = set()
            genes_set = set()
            path_genes = {t for t in pathway_tfs[pathway]}
            for tf in tfs:
                tf_genes = set(specific_g.neighbors(tf))
                in_genes = tf_genes.intersection(path_genes)
                others_genes = path_genes - in_genes
                tf_feat = feat_dict[tf]
                genes_set.update(in_genes)
                genes_set.update(others_genes)

                if len(in_genes) == 0:
                    continue
                for g2 in others_genes:
                    if g2 not in feat_dict.keys():
                        continue
                    for g1 in in_genes:
                        p_val = func1(tf_feat, feat_dict[g1], feat_dict[g2], permutation=permutation)
                        if p_val <= threshold:
                            out.update({g1, g2})
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

            d = feat.shape[1] // stage
            curr_exp = [feat_dict[g][i * d:(i + 1) * d] for g in genes]
            init_exp = [feat_dict[g][0:d] for g in genes]

            pvalues = permutation_test(curr_exp, init_exp)
            path_result[3] = pvalues
            new_results.append(path_result)

        pathway_tf_score = sorted(new_results, key=lambda x: x[3])
        dataframe = pd.DataFrame(pathway_tf_score, columns=["pathway", "nums", "percent", "pvalue"])
        save_df(dataframe, f"../out/{name}/pathway_score", f"factor{i + 1}.csv",
                header=["Pathway", "Nums", "Percent", "PValue"])


if __name__ == '__main__':
    name = "DTGN-16"
    annotation_path = "../data/IPSC/pathways_genes.csv"
    positive_pathway_path = "../data/IPSC/valid_data/positive_pathways.csv"
    negative_pathway_path = "../data/IPSC/valid_data/negative_pathways.csv"

    tf_pathway_score(name, annotation_path, positive_pathway_path, negative_pathway_path, stage=7, permutation=10)
