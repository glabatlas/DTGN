# @Author : CyIce
# @Time : 2024/6/26 15:41


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc

from utils.file_operation import save_df


def draw_roc(name, permutation_path, positive_tf_path, negative_tf_path, start_stage, end_stage, top=-1, compare=False):
    positive_tfs = pd.read_csv(positive_tf_path).iloc[:, 0].values.tolist()
    positive_tfs = {tf for tf in positive_tfs}
    negative_tfs = pd.read_csv(negative_tf_path).iloc[:, 0].values.tolist()
    negative_tfs = {tf for tf in negative_tfs}
    valid_tfs = positive_tfs.union(negative_tfs)

    tf_p_dict = {}
    for i in range(start_stage, end_stage):
        if compare:
            tf_list = pd.read_excel(f"{permutation_path}/factor{i + 1}.xlsx").iloc[:, 0].values.tolist()
        else:
            tf_list = pd.read_csv(f"{permutation_path}/stage{i + 1}.csv").iloc[:, 0].values.tolist()
        for tf in tf_list:
            tf = tf.split('_')[0]
            if tf in valid_tfs:
                tf_p_dict[tf] = []

    for i in range(start_stage, end_stage):
        if compare:
            tf_list = pd.read_excel(f"{permutation_path}/factor{i + 1}.xlsx").iloc[:, 0:2].values.tolist()
        else:
            tf_list = pd.read_csv(f"{permutation_path}/stage{i + 1}.csv").iloc[:, 0:2].values.tolist()
        if not tf_list:
            continue
        tf_list.sort(key=lambda x: x[1])

        rank_list = []
        rank = 1
        for j, (_, value) in enumerate(tf_list):
            if j > 0 and tf_list[j - 1][1] != value:
                rank += 1
            rank_list.append(rank)

        # update the rank of each TF.
        for j, rank in enumerate(rank_list):
            tf_list[j][1] = rank

        for tf, p in tf_list:
            tf = tf.split('_')[0]
            if tf in tf_p_dict.keys():
                tf_p_dict[tf].append(p)
        for k in tf_p_dict.keys():
            if len(tf_p_dict[k]) != i - start_stage + 1:
                tf_p_dict[k].append(len(tf_list) + 1)

    for k in tf_p_dict.keys():
        tf_p_dict[k] = np.mean(tf_p_dict[k])

    sorted_dict = sorted(tf_p_dict.items(), key=lambda x: x[1])
    result_keys = {key for key, value in sorted_dict[:top]}
    actual = []
    predict = []

    for tf in positive_tfs.union(negative_tfs):
        actual.append(1 if tf in positive_tfs else 0)
        if tf in tf_p_dict:
            predict.append(1 if tf in result_keys else 0)
        else:
            predict.append(0)

    tn, fp, fn, tp = confusion_matrix(actual, predict).ravel()

    tfs = sorted(tf_p_dict.keys(), key=lambda x: tf_p_dict[x])
    if top != -1:
        tfs = tfs[:top]

    y_true = []
    y_scores = []
    y_names = []
    for tf in tfs:
        if tf in positive_tfs:
            y_true.append(1)
        elif tf in negative_tfs:
            y_true.append(0)
        y_scores.append(round(tf_p_dict[tf], 4))
        y_names.append(tf)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precision_score = y_true.sum() / len(y_true)
    recall_score = y_true.sum() / len(positive_tfs)

    out_final_p = sorted(zip(y_names, y_scores, y_true), key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(out_final_p)
    save_df(df, f"../out/{name}", "final_p.csv", header=['TF', 'PValue', 'Label'])

    y_scores = -y_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
    auprc = round(auc(recall, precision), 3)
    auroc = round(roc_auc_score(y_true, y_scores), 3)
    f1_score = round(2 * (precision_score * recall_score) / (precision_score + recall_score), 4)

    print(f"{round(auroc, 4)}", end='\t')
    print(f"{round(precision_score, 4)}", end='\t')
    print(f"{round(recall_score, 4)}", end="\t")
    print(f"{round(f1_score, 4)}")

    plt.title(label=name)
    plt.plot(fpr, tpr)
    plt.show()


def test(name,dataset):
    compare = False
    permutation_path = f"../out/{name}/permutation"
    positive_tf_path = f"../data/{dataset}/valid_data/early_tf.csv"
    negative_tf_path = f"../data/{dataset}/valid_data/later_tf.csv"
    # positive_tf_path = f"../data/{dataset}/valid_data/positive_tfs.csv"
    # negative_tf_path = f"../data/{dataset}/valid_data/negative_tfs.csv"
    start_stage = 3
    end_stage = 6
    draw_roc(name, permutation_path, positive_tf_path, negative_tf_path, start_stage, end_stage, top=-1, compare=compare)


if __name__ == '__main__':
    for id in [1,2,4,6,8,12,16]:
        test(f"LR{id}", "LR")
