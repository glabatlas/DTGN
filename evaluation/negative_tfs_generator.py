# @Author : CyIce
# @Time : 2024/6/27 10:43


import pandas as pd

from utils.pathway_utils import read_pathway


def create_tfs(name, pos_tfs_path, pos_pathways_path, pathway_path, out_path, min_overlap):
    """
    create netative tfs according to the given positive tfs and known pathway-gene annotations.
    """
    pathway_dict = read_pathway(pathway_path)
    positive_genes = pd.read_csv(pos_tfs_path).iloc[:, 0].values.tolist()
    train_tfs = pd.read_csv(f"../data/{name}/network.csv").iloc[:, 0].values.tolist()
    positive_genes = {g for g in positive_genes}
    train_tfs = {g for g in train_tfs}

    positive_tfs = positive_genes.intersection(train_tfs)
    # positive_tfs = positive_genes

    all_pathways = {path for path in pathway_dict.keys()}
    positive_pathways = pd.read_csv(pos_pathways_path, sep='\t').iloc[:, 0].values
    positive_pathways = {p for p in positive_pathways}

    pos_pw_gene = set()
    pos_pw_gene.update(positive_tfs)
    for pw in positive_pathways:
        pos_pw_gene.update(pathway_dict[pw])

    for key in pathway_dict.keys():
        if (len(pos_pw_gene.intersection(pathway_dict[key])) > min_overlap):
            positive_pathways.add(key)

    negative_pathways = all_pathways - positive_pathways

    print(len(all_pathways), len(positive_pathways), len(negative_pathways))

    negative_gene = set()
    for path in negative_pathways:
        if path in pathway_dict.keys():
            negative_gene.update(pathway_dict[path])

    negative_tfs = negative_gene.intersection(train_tfs)
    print(f"All TFs: {len(train_tfs)}, Positive TFs: {len(positive_tfs)},Negative TFs: {len(negative_tfs)}")

    # '''
    pos_data = pd.DataFrame(positive_tfs, columns=["gene"])
    neg_data = pd.DataFrame(negative_tfs, columns=["gene"])
    pos_data.to_csv(f"{out_path}/positive_tfs.csv", index=False)
    neg_data.to_csv(f"{out_path}/negative_tfs.csv", index=False)
    # '''


if __name__ == '__main__':
    min_overlap = 4
    create_tfs("IPSC", "../history_code/IPSC/valid_data/related_tfs.csv",
               "../history_code/IPSC/valid_data/positive_pathways.csv",
               "../history_code/IPSC/pathways_genes.csv", "../history_code/IPSC/valid_data", min_overlap)
