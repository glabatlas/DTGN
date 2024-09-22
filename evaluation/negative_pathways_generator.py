# @Author : CyIce
# @Time : 2024/6/27 10:43

import pandas as pd

from utils.pathway_utils import read_pathway


def create_negative_pathways(name, positive_pathway_path, pathway_path, out_path, min_overlap):
    """
    create negative pathways according to the given positive pathways and known pathway-gene annotations
    """
    pathway_dict = read_pathway(pathway_path)
    positive_pathways = pd.read_csv(positive_pathway_path,sep='\t').iloc[:, 0].values.tolist()
    positive_pathways = {name for name in positive_pathways}
    negative_pathways = pathway_dict.keys() - positive_pathways

    related_gene_set = set()
    for name in positive_pathways:
        if name in pathway_dict.keys():
            related_gene_set.update(pathway_dict[name])

    while (True):
        remove_list = []
        for neg in negative_pathways:
            if len(related_gene_set.intersection(pathway_dict[neg])) > min_overlap:
                remove_list.append(neg)
        for name in remove_list:
            related_gene_set.update(pathway_dict[name])
            negative_pathways.remove(name)
        # break
        if len(remove_list) == 0:
            break

    print(len(positive_pathways), len(negative_pathways))

    dataframe = pd.DataFrame(negative_pathways, columns=['negative_pathway'])
    dataframe.to_csv(f"{out_path}/negative_pathways.csv", index=False)


if __name__ == '__main__':
    min_overlap = 4
    create_negative_pathways("IPSC", "../history_code/IPSC/valid_data/positive_pathways.csv",
                             "../history_code/IPSC/pathways_genes.csv", "../history_code/IPSC/valid_data", min_overlap)
