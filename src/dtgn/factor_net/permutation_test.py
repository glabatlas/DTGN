# @Author : CyIce
# @Time : 2024/6/25 19:22

import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.stats import multitest
from ..utils.file_operation import save_df


def permutation(list1, list2, permutation_times):
    """
    Using the permutation test to calculate the p-value for two gene expression list.
    """
    observed_difference = np.mean(list1) - np.mean(list2)
    combined_data = np.concatenate((list1, list2)).reshape(-1)
    permuted_differences = []
    n = combined_data.shape[0] // 2
    for _ in range(permutation_times):
        np.random.shuffle(combined_data)
        permuted_group1 = combined_data[:n]
        permuted_group2 = combined_data[n:]
        permuted_difference = np.mean(permuted_group1) - np.mean(permuted_group2)
        permuted_differences.append(permuted_difference)

    # Calculating the p-value using the permutation test
    p_value = (np.abs(permuted_differences) >= np.abs(observed_difference)).mean()
    return p_value


def diff_exp_test(name, num_stages,permutation_times, pvalue=0.01):
    """
    Calculates the difference in expression between the current stage and the initial stage for each transcription factor (TF).

    Parameters:
    name (str): The name of the dataset or experiment.
    num_stages (int): The number of stages to process.
    permutation_times (int): The number of permutations for statistical testing.
    pvalue (float, optional): The p-value threshold for filtering edges. Defaults to 0.01.

    Returns:
    All outputs are stored in the "./out/{name}/permutation" directory.

    Example:
    >>> diff_exp_test("experiment", 5, 1000)
    """
    feat = np.array(pd.read_csv(f'./out/{name}/features.csv').iloc[:, 1:].values)
    gene_id_dict = dict(zip(pd.read_csv(f'./out/{name}/features.csv').iloc[:, 0], range(len(feat))))
    dim = feat.shape[1] // num_stages

    for i in range(1, num_stages):
        edges = pd.read_csv(f"./out/{name}/dynamicTGNs/factor{i + 1}.csv").iloc[:, 0:3].values.tolist()
        edges = [[e[0], e[1]] for e in edges if e[2] <= pvalue]
        G = nx.Graph()
        G.add_edges_from(edges)
        tfs = {tf[0] for tf in edges}
        out_tfs, out_pvalue, out = [], [], []
        for tf in tfs:
            targets = {tg for tg in G.neighbors(tf)}
            targets_id = [gene_id_dict[tg] for tg in targets]
            curr_exp = feat[targets_id][:, i * dim:(i + 1) * dim]
            init_exp = feat[targets_id][:, 0:dim]
            pvalues = permutation(curr_exp, init_exp, permutation_times)
            out.append([tf, pvalues])

        out = sorted(out, key=lambda x: x[1])

        dataframe = pd.DataFrame(out)
        save_df(dataframe, f"./out/{name}/permutation", f"stage{i + 1}.csv", ["GeneSymbol", "PValue"])
