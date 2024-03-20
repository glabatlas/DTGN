import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.stats import multitest
from utils import file_operation


def permutation(list1, list2, times=10000):
    observed_difference = np.mean(list1) - np.mean(list2)
    combined_data = np.concatenate((list1, list2)).reshape(-1)
    permuted_differences = []
    n = combined_data.shape[0]//2
    for _ in range(times):
        np.random.shuffle(combined_data)
        permuted_group1 = combined_data[:n]
        permuted_group2 = combined_data[n:]
        permuted_difference = np.mean(permuted_group1) - np.mean(permuted_group2)
        permuted_differences.append(permuted_difference)

    # 计算P值
    p_value = (np.abs(permuted_differences) >= np.abs(observed_difference)).mean()
    return p_value


def diff_exp_test(dataset,stage=6, run_id="", using_exp=False, global_net=False, pvalue=0.01, times=10000, mapping_path='',
                  type="tf"):
    out_id = run_id
    if using_exp:
        feat = np.array(pd.read_csv(f'./out/{dataset}/features/exp/features.csv').values)
        out_id += '_exp'
    else:
        out_id += '_hidden'
        feat = np.array(pd.read_csv(f'./out/{dataset}/features/{run_id}/features.csv').values)
    mapping = np.array(pd.read_csv(mapping_path).iloc[:, 0:2].values)
    gene_id_dict = dict(zip(mapping[:, 1], mapping[:, 0]))
    G = nx.Graph()
    if global_net is True:
        out_id += '_global'
        global_edges = pd.read_csv(f"./data/{dataset}/origin_net.csv").iloc[:,
                       0:2].values.tolist()
        G.add_edges_from(global_edges)
    else:
        out_id += '_factor'
    for i in range(1, stage):
        edges = pd.read_csv(f"./out/{dataset}/factor_link/{run_id}/factor{i + 1}.csv").iloc[:,
                0:3].values.tolist()
        edges = [[e[0],e[1]] for e in edges if e[2] <= pvalue]
        if global_net is False:
            G = nx.Graph()
            G.add_edges_from(edges)
        if type == 'tf':
            tfs = {tf[0] for tf in edges}
            # tfs = {node for pair in edges for node in pair}.intersection(valid_set_tfs)
        else:
            tfs = {node for pair in edges for node in pair}
        out_tfs, out_pvalue, out = [], [], []
        for tf in tfs:
            targets = {tg for tg in G.neighbors(tf)}
            targets_id = [gene_id_dict[tg] for tg in targets]
            if using_exp:
                curr_exp = feat[targets_id][:, i]
                init_exp = feat[targets_id][:, 0]
            else:
                d = feat.shape[1]//stage
                curr_exp = feat[targets_id][:, i*d:(i+1)*d]
                init_exp = feat[targets_id][:, 0:d]
            pvalues = permutation(curr_exp, init_exp, times=times)
            out.append([tf, pvalues])

        out = sorted(out, key=lambda x: x[1])
        out_id = out_id.replace("0.01",str(pvalue))
        file_operation.save_features(path=f"./out/{dataset}/permutation/{out_id}",
                                     filename=f'factor{i + 1}.csv', feat=out,
                                     columns=['tf', 'p-value'])

