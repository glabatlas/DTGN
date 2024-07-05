import pandas as pd

from evaluation.pathway_identify import read_pathway

def create():

    pass

organism = "Mus"
dataset = "MK"

pathway_dict = read_pathway(path=f"../data2/pathway/{dataset}/all.txt")

positive_genes = pd.read_csv(f"../data2/{organism}_{dataset}_cellnet/valid_data/all_tf.csv").iloc[:, 0].values.tolist()
# total_nums = len(positive_genes)
# positive_genes = positive_genes[0:int(total_nums/3)]

train_tfs = pd.read_csv(f"../data2/{organism}_{dataset}_cellnet/origin_net.csv").iloc[:, 0].values.tolist()

positive_genes = {g for g in positive_genes}
train_tfs = {g for g in train_tfs}

positive_tfs = positive_genes.intersection(train_tfs)
# positive_tfs = positive_genes
print(f"Positive TFs: {len(positive_tfs)}")

all_pathways = {path for path in pathway_dict.keys()}
positive_pathways = pd.read_csv(f"../data2/pathway/{dataset}/positive_pathways.csv", sep='\t').iloc[:, 0].values
positive_pathways = {p for p in positive_pathways}
# positive_pathways = set()
negative_pathways = set()
# negative_pathways = pd.read_csv(f"../data/pathway/{dataset}/negative_pathways.csv",sep='\t').iloc[:, 0].values
for key in pathway_dict.keys():
    if (len(positive_tfs.intersection(pathway_dict[key])) > 0):
        positive_pathways.add(key)

negative_pathways = all_pathways - positive_pathways
print(len(all_pathways), len(positive_pathways), len(negative_pathways))

negative_gene = set()
for path in negative_pathways:
    if path in pathway_dict.keys():
        negative_gene.update(pathway_dict[path])

negative_tfs = negative_gene.intersection(train_tfs)
print(f"Negative TFs: {len(negative_tfs)}")

# '''
pos_data = pd.DataFrame(positive_tfs, columns=["gene"])
neg_data = pd.DataFrame(negative_tfs, columns=["gene"])
pos_data.to_csv(f"../data/{organism}_{dataset}_cellnet/valid_data/positive_tfs.csv", index=False)
neg_data.to_csv(f"../data/{organism}_{dataset}_cellnet/valid_data/negative_tfs.csv", index=False)
# '''
