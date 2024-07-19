import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from pathos.pools import ProcessPool
import argparse

from utils.TGMI import triple_interaction, discretize
from utils.file_operation import save_df
from utils.pathway_utils import read_pathway

"""
This script is used to calculate the pathway-TF score.
"""

def time_specific_network(path):
    """
    Get the time-specific network.
    """
    edges = pd.read_csv(path).iloc[:, 0:2].values.tolist()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def time_specific_tf(path, p=0.05):
    """
    Get the significant time-specific tfs.
    """
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

    # calculate the p values using the permutation test.
    p_value = (np.abs(permuted_differences) >= np.abs(observed_difference)).mean()
    return p_value


def tf_pathway_score(dataset, name, pathway_path, nums_stage=10, threshold=0.05, permutation_times=1000):
    pathway_tfs = read_pathway(pathway_path)
    pathway_name = [name for name in pathway_tfs.keys()]

    feat = pd.read_csv(f"../out/{name}/features.csv")
    gene_id_dict = dict(zip(feat.iloc[:, 0], range(len(feat))))
    feat = np.array(feat.iloc[:, 1:].values)

    for i in range(1, nums_stage):
        print(f"stage: {i}")
        specific_tfs = time_specific_tf(f"../out/{name}/permutation/stage{i + 1}.csv", p=threshold)

        specific_g = time_specific_network(f"../out/{name}/dynamicTGNs/factor{i + 1}.csv")
        specific_feat = pd.read_csv(f"../out/{name}/features.csv")
        specific_feat = dict(zip(specific_feat.iloc[:, 0].values.tolist(), specific_feat.iloc[:, 1:].values.tolist()))

        for k in specific_feat.keys():
            specific_feat[k] = discretize(specific_feat[k])

        def per_pathway(pathway, tfs, func1, permutation):
            out = set()
            genes_set = set()
            path_genes = {t for t in pathway_tfs[pathway]}
            for tf in tfs:
                if tf not in specific_g.nodes():
                    continue
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
                        if g1 not in specific_feat.keys():
                            continue
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
                                  permutation=permutation_times,
                                  func1=triple_interaction)
        pool = ProcessPool(16)
        results = list(pool.amap(partial_wrapper, pathway_name).get())
        new_results = []
        for path_result in results:
            if path_result[2] == 0:
                continue
            genes = list(path_result[3])
            targets_id = [gene_id_dict[tg] for tg in genes]

            d = feat.shape[1] // nums_stage
            curr_exp = feat[targets_id][:, i * d:(i + 1) * d]
            init_exp = feat[targets_id][:, 0:d]

            pvalues = permutation_test(curr_exp, init_exp)
            path_result[3] = pvalues
            new_results.append(path_result)

        pathway_tf_score = sorted(new_results, key=lambda x: x[3])
        dataframe = pd.DataFrame(pathway_tf_score, columns=["pathway", "nums", "percent", "pvalue"])
        save_df(dataframe, f"../out/{name}/pathway_score", f"factor{i + 1}.csv",header=["pathway", "nums", "percent", "pvalue"])


def args_parsing():
    parser = argparse.ArgumentParser(description='Parsing args on main.py')
    parser.add_argument('--dataset', type=str, default="iPSC")
    parser.add_argument('--nums_stage', type=int, default=7)
    parser.add_argument('--pathway_path', type=str, default="../data/IPSC/pathways_genes.csv")
    parser.add_argument('--permutation_times', type=int, default=200)
    parser.add_argument('--name', type=str, default="iPSC-SSN")
    parser.add_argument('--threshold', type=float, default=0.05)
    return parser


if __name__ == '__main__':
    main_parser = args_parsing()
    args = main_parser.parse_args()
    dataset = args.dataset
    pathway_path = args.pathway_path
    nums_stage = args.nums_stage
    name = args.name
    permutation_times = args.permutation_times
    threshold = args.threshold

    tf_pathway_score(dataset, name, pathway_path=pathway_path, nums_stage=nums_stage, threshold=threshold,
                     permutation_times=permutation_times)
