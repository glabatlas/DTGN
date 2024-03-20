import pandas as pd
import numpy as np
import os


def save_features(path, filename, feat, columns=None):
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe = pd.DataFrame(feat, columns=columns)
    dataframe.to_csv(path + '/' + filename, index=False)
    print(f"saved features to {path + '/' + filename}")


def load_feature(path, epoch=1):
    feat = np.array(pd.read_csv(path + f'features{epoch}.csv').values)
    return feat


def read_genes(path, stage=6, type="node"):
    out = []
    database_id = 4
    for i in range(1, stage + 1):
        edges = pd.read_csv(path + f"factor{i}.csv").iloc[:, 0:2].values.tolist()
        # edges = pd.read_csv(path + f"mark_genes{database_id}-{i}.csv").iloc[:, 0:2].values.tolist()
        if type == "node":
            nodes = {node for pair in edges for node in pair}
            out.append(nodes)
        elif type == "edge":
            # edges = {u + '-' + v for u, v in edges}
            # edges = {u + '-' + v for u, v in edges if u<=v else v+'-'+u}
            edges = {u + '-' + v if u <= v else v + '-' + u for u, v in edges}
            out.append(edges)
        else:
            NotImplemented
    return out
