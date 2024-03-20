Introduction
--------------------------------------

We developed DyTGNets, a method to identify phenotype-specific transcriptional elements based on time-course gene expression data and TF-Gene network. 
This method uses the graph autoencoder (GAE) model to learn the temporal latent representation from time-course gene expression data and TF-Gene network,
and then employs the extended sample-specific network (SSN) method to construct dynamic TF-Gene networks. 
Finally, we utilize the dynamic TF-Gene networks to identify phenotype-specific transcriptional elements.

Dependencies
-------------------------------------
python 3.8;   
pytorch 2.0.1;   
scikit-learn 1.2.2;   
pyg 2.3.1;   
networkx 3.1;   
numpy 1.24.3;
pandas 1.5.3.

Installation
-------------------------------------
git this project directory

cd ./DyTGNets/


How to use
--------------------------------------
### Data processing
`preprocessing.py`: The time-course gene expression data and TF-Gene network will be filtered based on their mean and variance.

input: data/dataset/raw_data/exp.csv, data/dataset/raw_data/network.csv, data/dataset/raw_data/mapping.csv

output: data/dataset/training_node.csv, data/dataset/training_net.csv, data/dataset/mapping.csv

```python
>> python preprocessing.py -dataset LP2 -stage 10 -mean 1 -var 0 
```


### Trainning the DyTGNets model to construct dynamic TF-Gene networks
`main.py`: Trainning or using the DyTGNets model to construct dynamic TF-Gene networks.
input: data/dataset/training_node.csv, data/dataset/training_net.csv, data/dataset/mapping.csv

output: out/dataset/factor_link, out/dataset/model, out/dataset/permutation

Trainning a new model:
```python
>> python main.py -dataset LP -method DyTGNets -train true -stage 10 -epoch 30000
```
Loading the trained model:
```python
>> python main.py -dataset LP -method DyTGNets -train false -stage 10
```

### Identifying phenotype-specific pathways
`pathway_identify.py`  To identify phenotype-specific pathways using dynamic TF-Gene networks.

input: data/pathway/dataset/all.txt, data/pathway/dataset/positive_pathways.csv, data/pathway/dataset/negative_pathways.csv, out/dataset/permutation/DyTGNets_hidden_factor/factor.csv, out/dataset/features/DyTGNets/features.csv

output: out/dataset/pathway_score/DyTGNets/factor.csv
```python
>> python pathway_identify.py -dataset LP -method DyTGNets -stage 10 -times 1000 -threshold 0.01
```