1\. Introduction
--------------------------------------

We developed DTGN, a method to identify phenotype-specific transcriptional elements based on time-course gene expression data and TF-Gene network. 
This method uses the graph autoencoder (GAE) model to learn the temporal latent representation from time-course gene expression data and TF-Gene network,
and then employs the extended sample-specific network (SSN) method to construct dynamic TF-Gene networks. 
Finally, we utilize the dynamic TF-Gene networks to identify phenotype-specific transcriptional elements.

2\. Dependencies
-------------------------------------
The following are the necessary dependencies and recommended versions. The version of Python depends on your CUDA version and PyTorch version. Please refer to: https://pytorch.org/get-started/locally/


    python>=3.8;   
    pytorch 2.0.1;   
    scikit-learn 1.2.2;   
    pyg 2.3.1;   
    networkx 3.1;   
    scipy>=1.10;   
    tqdm>=4.64;   
    matplotlib>=3.7;   
    numpy 1.24.3;   
    pandas 1.5.3.

3\. Usage
--------------------------------------
## 3.1 Installation

You can use the DTGN model in two ways:
1. Clone this repository to your working directory.
2. Download the [dtgn.whl](https://github.com/glabatlas/DTGN/tree/main/dist/dtgn-1.0.0-py3-none-any.whl)
 or [dtgn.tar.gz](https://github.com/glabatlas/DTGN/tree/main/dist/dtgn-1.0.0.tar.gz) and install via pip: `pip install DTGN.whl` or `pip install DTGN.tar.gz`


## 3.2 Usage Example

### 3.2.1 Using the python package

- Step 1: Download the [dtgn.whl](https://github.com/glabatlas/DTGN/tree/main/dist/dtgn-1.0.0-py3-none-any.whl)
 or [dtgn.tar.gz](https://github.com/glabatlas/DTGN/tree/main/dist/dtgn-1.0.0.tar.gz) and install via pip: 
  ```bash
  pip install DTGN.whl 
  ```
  or
  ```bash
  pip install DTGN.tar.gz
  ```

- Step 2: Download the data from [data](https://github.com/glabatlas/DTGN/tree/main/data) and [out](https://github.com/glabatlas/DTGN/tree/main/out), then add them to the project directory.

- Step 3: Train or load the model
  - Train a new model
  ```bash
  dtgn-train --name LR --num_stages 10 --train true --epochs 3000 --exp_path ./data/LR/exp.csv --net_path ./data/LR/network.csv --encoder_layer 16,8,2 --decoder_layer 2,8,16
  ```
  - Load the exist model
  ```bash
  dtgn-train --name LR --num_stages 10 --train false --exp_path ./data/LR/exp.csv --net_path ./data/LR/network.csv --encoder_layer 16,8,2 --decoder_layer 2,8,16
  ```
  - There are several primary parameters. For details, use `dtgn-train --help`.
    - **--name**: The model name and output directory.
    - **--exp_path**: The path to the gene expression data.
    - **--net_path**: The path to the TF-Gene network data.
    - **--num_stages**: The number of stages.
    - **--epochs**: The number of training epochs.
    - **--encoder_layer**: The encoder layer size.
    - **--decoder_layer**: The decoder layer size.
    - For different datasets, the number of stages, encoder layers, and decoder layers vary. The first parameter of the encoder and the last parameter of the decoder remain consistent, representing the size of the one-hot encoding dimension. Details are listed below:
  
      |     | num_stages | encoder_layer | decoder_layer |
      |-----|------------|---------------|---------------|
      | LR  | 10         | 16,8,2        | 2,8,16        |
      | MI  | 6          | 32,8,2        | 2,8,32        |
      | HCV | 5          | 24,8,2        | 2,8,24        |

- Step 4: Predict the phenotype-specific TFs in each stage
  ```bash
  dtgn-tf --name LR --num_stages 10 --permutation_times 10000
  ```
  The output files are stored in the "./out/name/" directory.
  
- Step 5: Predict the phenotype-specific pathways in each stage
  ```bash
  dtgn-pw --dataset LR --name LR --num_stages 10 --permutation_times 2000 --threshold 0.01 --pathway_path "./data/LR/pathways_genes.csv"
  ```
  The output files are stored in the "./out/name/" directory.



### 3.2.2 Using command line
**input data**: The time course gene expression data and TF-Gene network data. Please ensure that the data format is consistent with the description above.

- Step 1: git this project directory

- Step 2: cd ./DTGN/

- Step 3: run the command as follows:

  - Train a new model
  ```bash
  python main.py --name LR --train true --exp_path ./data/LR/exp.csv --net_path ./data/LR/network.csv --encoder_layer 16,8,2 --decoder_layer 2,8,16
  ```

  - Load the exist model
  ```bash
  python main.py --name LR --train false --exp_path ./data/LR/exp.csv --net_path ./data/LR/network.csv --encoder_layer 16,8,2 --decoder_layer 2,8,16
  ```
  - There are several primary parameters. For details, use `python main.py --help`.
    - **--name**: The model name and output directory.
    - **--exp_path**: The path to the gene expression data.
    - **--net_path**: The path to the TF-Gene network data.
    - **--num_stages**: The number of stages.
    - **--epochs**: The number of training epochs.
    - **--encoder_layer**: The encoder layer size.
    - **--decoder_layer**: The decoder layer size.
    - For different datasets, the number of stages, encoder layers, and decoder layers vary. The first parameter of the encoder and the last parameter of the decoder remain consistent, representing the size of the one-hot encoding dimension. Details are listed below:
  
      |     | num_stages | encoder_layer | decoder_layer |
      |-----|------------|---------------|---------------|
      | LR  | 10         | 16,8,2        | 2,8,16        |
      | MI  | 6          | 32,8,2        | 2,8,32        |
      | HCV | 5          | 24,8,2        | 2,8,24        |

- Step 4: The output files are stored in the "./out/name/" directory.

## 3.3 Trainnig Data Format

### 3.3.1 Gene Expression Data

- **Header Row**: The first row contains the column headers. The first column is geneSymbol, and the subsequent columns are labeled as time1, time2, ..., timeN.
- **Data Rows**: Each subsequent row represents the expression data for a specific gene at different time points.

  | geneSymbol | time1 | time2 | time3 | time4 | ... | timeN |
  |------------|-------|-------|-------|-------|-----|-------|
  | g1         | 2.3   | 2.5   | 2.7   | 2.8   | ... | 3.0   |
  | g2         | 1.1   | 1.3   | 1.4   | 1.5   | ... | 1.6   |
  | g3         | 0.5   | 0.6   | 0.7   | 0.8   | ... | 0.9   |
  | g4         | 3.2   | 3.3   | 3.5   | 3.6   | ... | 3.8   |
  | ...        | ...   | ...   | ...   | ...   | ... | ...   |
  | gN         | 4.1   | 4.2   | 4.3   | 4.4   | ... | 4.5   |

### 3.3.2 TF-Gene Network Data

- **Header Row**: The first row contains the column headers: source and target.
- **Data Rows**: Each subsequent row represents an edge in the network.

  | source | target |
  |--------|--------|
  | Phf5a  | Fgf1   |
  | Phf5a  | Nrbp2  |
  | Phf5a  | Kat2b  |
  | g1     | g2     |
  | g2     | g3     |
  | ...    | ...    |


4\. API Documentation
--------------------------------------
More detailed information can be found in the documentation comments of the method.

## 4.1 dtgn.preprocessing(name, exp_path, net_path, mean, var, norm_type='id')
Filter gene expression based on mean and variance to ensure network connectivity.

### Parameters

- **name (str)**: The name of the dataset or experiment for saving the model and output data.

- **exp_path (str)**: The file path to the gene expression data.

- **net_path (str)**: The file path to the TF-Gene network data.

- **mean (float)**: The mean value used for filtering.

- **var (float)**: The variance value used for filtering.

- **norm_type (str, optional)**: The type of normalization to apply. Defaults to `'id'`. Options include:
    - `'id'`: No normalization.
    - `'zscore'`: Z-score normalization.
    - `'minmax'`: Min-max scaling.

### Returns:

- **tuple**: A tuple containing the processed expression data and network edges.

### Example:

```python
exp, edges = preprocessing("experiment", "./data/LR/exp.csv", "./data/LR/network.csv", 0.5, 0.1, norm_type='id')
```

## 4.2 dtgn.one_hot_encode(feat, num_intervals, origin_val=False)
Applies one-hot encoding to represent gene expression data.

### Parameters

- **feat (torch.Tensor)**: A tensor containing the gene expression data with shape (T, N, 1).
- **num_intervals (int)**: The number of intervals for encoding.
- **origin_val (bool, optional)**: If True, use the original values for encoding; otherwise, use binary encoding. Defaults to False.

### Returns:
- **tuple**: A tuple containing:
        - one_hot_feat (torch.Tensor): The one-hot encoded feature tensor with shape (stage_nums, n, num_intervals).
        - one_hot_pos (torch.Tensor): A tensor of interval indices with shape (stage_nums * n).

### Example:
```python
feat = torch.tensor([[[0.1], [0.5]], [[0.2], [0.8]]])
one_hot_feat, one_hot_pos = one_hot_encode(feat, 5)
```

## 4.3 dtgn.MyGAE(encoder, decoder)
The DTGN model, a Graph Autoencoder (GAE) for gene expression data.

### Parameters
- **encoder (Module)**: The encoder neural network module.
- **decoder (Module)**: The decoder neural network module.

### Example:
```python
encoder = GCNEncoder([64, 32, 16])
decoder = GCNDecoder([16, 32, 64])
model = MyGAE(encoder, decoder)
loss = model.total_loss(feat, z, edges)
```

## 4.4 dtgn.GCNEncoder(hidden_list, activation=nn.Sigmoid())
Graph Convolutional Network (GCN) Encoder.

### Parameters
- **hidden_list (list of int)**: A list specifying the number of units in each hidden layer.
- **activation (callable, optional)**: The activation function to apply after each layer except the last. Defaults to nn.Sigmoid().

### Example:
```python
encoder = GCNEncoder([64, 32, 16])
output = encoder(features, edge_index)
```

## 4.5 dtgn.GCNDecoder(hidden_list, activation=nn.Sigmoid())
Graph Convolutional Network (GCN) Decoder.

### Parameters
- **hidden_list (list of int)**: A list specifying the number of units in each hidden layer.
- **activation (callable, optional)**: The activation function to apply after each layer except the last. Defaults to nn.ReLU().

### Example:
```python
decoder = GCNDecoder([16, 32, 64])
output = decoder(features, edge_index)
```

## 4.6 dtgn.get_factor_grn(name, feats, edges, idx2sybol, stage, threshold)
Constructs the dynamic TF-Gene network for each stage.

### Parameters
- **name (str)**: The name of the dataset or experiment.
- **feats (Tensor)**: The features for each node.
- **edges (Tensor)**: The edge indices.
- **idx2sybol (dict)**: A mapping from index to symbol.
- **stage (int)**: The number of stages to process.
- **threshold (float)**: The threshold for filtering edges.

### Returns:
- All outputs are saved in the "./out/{name}/dynamicTGNs" directory.

### Example:
```python
get_factor_grn("experiment", feats, edges, idx2sybol, 5, 0.05)
```

## 4.7 dtgn.train_pyg_gcn(name, genes, feat,edges, activation, lr, wd, epochs, device, encoder_layer, decoder_layer, is_train)
Trains or loads DTGN model using PyTorch Geometric.

### Parameters
- **name (str)**: The name of the dataset or experiment.
- **genes (list of str)**: List of gene symbols.
- **feat (Tensor)**: The feature matrix.
- **edges (list of tuples)**: List of edge connections.
- **activation (callable)**: The activation function to use in the model.
- **lr (float)**: Learning rate for the optimizer.
- **wd (float)**: Weight decay for the optimizer.
- **epochs (int)**: Number of training epochs.
- **device (str)**: Device to run the model on ('cpu' or 'gpu').
- **encoder_layer (list of int)**: Number of units in each layer of the encoder.
- **decoder_layer (list of int)**: Number of units in each layer of the decoder.
- **is_train (bool)**: Flag to indicate whether to train a new model or load an existing one.

### Returns:
- np.ndarray: The hidden features extracted by the model.

### Example:
```python
hidden_feats = train_pyg_gcn("experiment", gene_list, features, edge_list, nn.ReLU(), 0.01, 0.0001, 100, 'gpu', [64, 32], [32, 64], True)
```

5\. Project Structure
--------------------------------------
- **`main.py`**: The main entry point of the application.
- **`train.py`**: The function to train the model.
- **`arg_parser.py`**: To parse the command line arguments. 

- **`model/`**: Contains the source code of the DTGN model.
  - **`PygGCN.py`**: The encoder and decoder of the DTGN model.
  - **`MyGAE.py`**: The DTGN model.

- **`factor_net/`**: Contains the source code of the factor network and permutation methods.
  - **`factor_grn.py`**: To construct the dynamic TF-Gene network.
  - **`permutation.py`**: The function for determining the statistical significance of transcription factors at each time point.

- **`preprocess/`**: Contains the source code for preprocessing the data.
  - **`preprocessing.py`**: The function to preprocess the data to filter the data by mean and variance.

- **`evaluation/`**: Contains the source code of the negative sampling method and the calculation of the Receiver Operating Characteristic (ROC) curve.
  - **`negative_tfs_generator.py`**: To sample negative TFs for a specific phenotype.
  - **`negative_pathways_generator.py`**: To sample negative pathways for a specific phenotype.
  - **`pathway_identify.py`**: To calculate the score of each pathway according to TGMI.
  - **`pathways_roc.py`**: The function for calculating the Receiver Operating Characteristic (ROC) curve for pathways.
  - **`tfs_roc.py`**: The function for calculating the Receiver Operating Characteristic (ROC) curve for tfs.

- **`data/`**: Contains the data used in the DTGN model.
  - **`LR/`**: The data used in the LR dataset.
    - **`valid_data/`**: The validation data.
    - **`exp.csv`**: The time-course gene expression data.
    - **`network.csv`**: The TF-gene network.
    - **`pathways_genes.csv`**: The annotation of pathways.

- **`out/`**: Contains the output files of the DTGN model.
  - **`LR/`**: The output data for the LR dataset.
    - **`dynamicTGNs/`**: The dynamic TF-Gene networks in each time point.
    - **`permutation/`**: The permutation scores of the tfs in each time point.
    - **`pathway_score/`**: The significant scores of the pathways in each time point.
    - **`train_set/`**: The training data after filtering.
    - **`features.csv`**: The final temporal latent representation for each node at each time point.
    - **`final_p.csv`**: The final score of the tfs associated with each phenotype.
    - **`final_pathway_p.csv`**: The final score of the pathway associated with each phenotype.
    - **`model.pth`**: The trained model containing the model structure and parameters.

- **`utils/`**: Contains the utility functions used in the project.
- **`dist/`**: The installation files.

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`README.md`**: The README file you are currently reading.
- **`LICENSE`**: The license file for the project.
