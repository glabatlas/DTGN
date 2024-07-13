import pandas as pd

from evaluation.pathway_identify import read_pathway

dataset = "HCV"
pathway_dict = read_pathway(path=f"../data/pathway/{dataset}/all.txt",use_all=True)
# pathway_dict = read_pathway()
positive_pathways = pd.read_csv(f"../data/pathway/{dataset}/positive_pathways.csv").iloc[:, 0].values.tolist()
positive_pathways = {name for name in positive_pathways}
negative_pathways = pathway_dict.keys() - positive_pathways

related_gene_set = set()
for name in positive_pathways:
    if name in pathway_dict.keys():
        related_gene_set.update(pathway_dict[name])

while (True):
    nums = len(negative_pathways)
    remove_list = []
    for neg in negative_pathways:
        if len(related_gene_set.intersection(pathway_dict[neg])) > 4:
            remove_list.append(neg)
    for name in remove_list:
        related_gene_set.update(pathway_dict[name])
        negative_pathways.remove(name)
    # break
    if len(remove_list) == 0:
        break

print(len(negative_pathways))

dataframe = pd.DataFrame(negative_pathways, columns=['negative_pathway'])
dataframe.to_csv(f"../data/pathway/{dataset}/negative_pathways.csv", index=False)



