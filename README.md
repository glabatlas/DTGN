Introduction
--------------------------------------

We developed DTGN, a method to identify phenotype-specific transcriptional elements based on time-course gene expression data and TF-Gene network. 
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
numpy 1.24.3;<br>
pandas 1.5.3.


Project Structure
-------------------------------------

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

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`README.md`**: The README file you are currently reading.
- **`LICENSE`**: The license file for the project.




How to use
--------------------------------------
## Trainnig Data Format

### Gene Expression Data

- Header Row: The first row contains the column headers. The first column is geneSymbol, and the subsequent columns are labeled as TimePoint1, TimePoint2, ..., TimePointN.
- Data Rows: Each subsequent row represents the expression data for a specific gene at different time points.

| geneSymbol | TimePoint1 | TimePoint2 | TimePoint3 | TimePoint4 | ... | TimePointN |
|------------|-------------|-------------|-------------|-------------|-----|-------------|
| GeneA      | 2.3         | 2.5         | 2.7         | 2.8         | ... | 3.0         |
| GeneB      | 1.1         | 1.3         | 1.4         | 1.5         | ... | 1.6         |
| GeneC      | 0.5         | 0.6         | 0.7         | 0.8         | ... | 0.9         |
| GeneD      | 3.2         | 3.3         | 3.5         | 3.6         | ... | 3.8         |
| ...        | ...         | ...         | ...         | ...         | ... | ...         |
| GeneN      | 4.1         | 4.2         | 4.3         | 4.4         | ... | 4.5         |

### TF-Gene Network Data

- Header Row: The first row contains the column headers: source and target.
- Data Rows: Each subsequent row represents an edge in the network.

| source | target |
|--------|--------|
| Phf5a  | Fgf1   |
| Phf5a  | Nrbp2  |
| Phf5a  | Kat2b  |
| GeneA  | GeneB  |
| GeneC  | GeneD  |
| ...    | ...    |


## Eaxmple

git this project directory
cd ./DTGN/

```
python main.py --name LR --train true --exp_path ./data/LR/exp.csv --net_path ./data/LR/network.csv --mean 0 --var 0 --encoder_layer 32,8,2 --encoder_layer 2,8,32
```
###
- **--exp_path**: The path to the gene expression data.
- **--net_path**: THe path to the TF-Gene network data.
