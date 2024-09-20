# @Author : CyIce
# @Time : 2024/7/12 16:13


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from utils.draw_roc import draw_roc


def pathway_roc(dataset, name, stage, type="min"):
    pathway_score_dict = {}
    all_pathways_set = pd.read_csv(f"../data/{dataset}/pathways_genes.csv", sep="\t").iloc[:, 0].values.tolist()
    all_pathways_set = {name for name in all_pathways_set}
    pathway_size = len(all_pathways_set)
    for pw in all_pathways_set:
        pathway_score_dict[pw] = [pathway_size] * (stage - 1)
    for i in range(1, stage):
        pathway_score = pd.read_csv(f"../out/{name}/pathway_score/factor{i + 1}.csv").iloc[:, 0:4]

        pathway_score['rank'] = pathway_score['pvalue'].rank(method='min')
        last_rank = 0
        next_rank = 1
        k = 1
        for pathway, tf, percent, pvalue, rank in pathway_score.values:
            if pd.isna(pvalue) or pd.isna(pathway):
                continue
            if rank != last_rank:
                next_rank = k
                last_rank = rank
            pathway_score_dict[pathway][i - 1] = next_rank
            k += 1

    for k in pathway_score_dict.keys():
        if type == "min":
            pathway_score_dict[k] = np.min(pathway_score_dict[k])
        else:
            pathway_score_dict[k] = np.mean(pathway_score_dict[k])

    positive_pathway = pd.read_csv(f"../data/{dataset}/valid_data/positive_pathways.csv", sep='\t').iloc[:,
                       0].values
    positive_pathway = {p for p in positive_pathway}
    negative_pathway = pd.read_csv(f"../data/{dataset}/valid_data/negative_pathways.csv", sep='\t').iloc[:,
                       0].values
    negative_pathway = {p for p in negative_pathway}

    y_true = []
    y_scores = []
    y_term = []
    positive_scores = []
    negative_scores = []

    out_dict = {}
    for pw in pathway_score_dict.keys():
        if pw in positive_pathway:
            out_dict[pw] = [1, pathway_score_dict[pw]]
            positive_scores.append([pw, pathway_score_dict[pw]])
        elif pw in negative_pathway:
            out_dict[pw] = [0, pathway_score_dict[pw]]
            negative_scores.append([pw, pathway_score_dict[pw]])

    for k in out_dict.keys():
        y_true.append(out_dict[k][0])
        y_scores.append(out_dict[k][1])
        y_term.append(k)
    out_final_p = list(zip(y_true, [s for s in y_scores], y_term))
    out_final_p = sorted(out_final_p, key=lambda x: x[1])
    df = pd.DataFrame(out_final_p, columns=['label', 'value', 'term'])
    df.to_csv(f"../out/{name}/final_pathway_p.csv", index=False)

    y_true = np.array(y_true)
    y_scores = 1 - np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    auroc = round(roc_auc_score(y_true, y_scores), 3)

    return fpr, tpr, auroc


if __name__ == '__main__':
    dataset = "LR"
    draw_data = {}

    fpr, tpr, auroc = pathway_roc(dataset, "LR", stage=7, type="min")
    draw_data[f'DTGN: AUC={auroc}'] = (fpr, tpr)

    draw_roc(draw_data, title=f"{dataset} KEGG ROC", smooth=False,
             path=f"../out/img/KEGG.png")
