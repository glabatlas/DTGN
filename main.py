# @Author : CyIce
# @Time : 2024/6/25 10:13

import random
import os
import numpy as np
import pandas as pd

import torch
import dgl
from arg_parser import args_parsing
from preprocess.preprocessing import preprocessing
from utils.one_hot_encode import one_hot_encode
from model.MyGAE import MyGAE
from model.PygGCN import GCNEncoder, GCNDecoder
from factor_net.factor_grn import get_factor_grn
from factor_net.permutation_test import diff_exp_test
from train import train


def seed_everything(seed):
    """
    Setting the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()


def create_network(link_pairs, num_nodes, bi_directed=True, add_loop=False):
    """
    Create a dgl graph from a list of link pairs.
    """
    g = dgl.graph(link_pairs, num_nodes=num_nodes)
    g = dgl.remove_self_loop(g)
    if bi_directed:
        g = dgl.to_bidirected(g)
    if add_loop:
        g = dgl.add_self_loop(g)
    return g


def train_pyg_gcn(name, genes, feat, activation, lr, wd, epochs, device, encoder_layer, decoder_layer):
    train_edges = torch.tensor(edges).T
    train_feat = feat.T.unsqueeze(-1)
    train_feat, one_hot_pos = one_hot_encode(train_feat, encoder_layer[0])

    model = MyGAE(GCNEncoder(encoder_layer, activation=activation), GCNDecoder(decoder_layer, activation=activation))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train(name, model, genes, train_feat, train_edges, optim, device, epochs)
    torch.save(model, f"./out/{name}/model.pth")

    # model = torch.load(f"out/{name}/model.pth", map_location=device)
    # hidden_feat = model.encoder(train_feat, train_edges).permute(1, 0, 2).reshape((train_feat.size(1), -1))
    # data = pd.DataFrame( hidden_feat.detach().numpy())
    # print(train_feat.shape)
    # data.to_csv(f'out/{name}/features.csv', index=False)
    # exit(0)


if __name__ == '__main__':
    args = args_parsing().parse_args()
    seed_everything(42)

    is_train = args.train
    activation = torch.nn.Sigmoid() if args.activation == 'sigmoid' else torch.nn.ReLU()
    device = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")
    encoder_layer = list(map(int, args.encoder_layer.split(',')))
    decoder_layer = list(map(int, args.decoder_layer.split(',')))

    # Create the output folder if it does not exist.
    if not os.path.exists(f"./out/{args.name}"):
        os.makedirs(f"./out/{args.name}")

    exp, edges = preprocessing(args.exp_path, args.net_path, args.mean, args.var, args.norm_type)
    genes = [row[0] for row in exp]
    feats = np.array([row[1:] for row in exp])
    feats = torch.tensor(feats).squeeze()
    num_stages = feats.shape[1]

    # create the mapping beween symbol and index
    sybol2idx = {row[0]: index for index, row in enumerate(exp)}
    idx2sybal = {idx: sybol for sybol, idx in sybol2idx.items()}
    # convert symbol to index
    edges = [[sybol2idx[edge[0]], sybol2idx[edge[1]]] for edge in edges]
    # create the dgl graph
    num_nodes = len(genes)
    g = create_network(edges, num_nodes)
    print(f"TF-Gene network: {g}")

    # training DTGN model
    if args.train == "true":
        print("Training...")
        train_pyg_gcn(args.name, genes, feats, activation, args.lr, args.wd, args.epochs, device, encoder_layer,
                      decoder_layer)
    else:
        print("train == false !")

    # Constructing the dynamic TF-Gene network for each stage.
    print("Constructing dynamic TGNs...")
    get_factor_grn(args.name, feats, edges, idx2sybal, num_stages, args.threshold)

    # Using permutation test to test the significance of the TFs.
    print("Permutation test...")
    diff_exp_test(args.name, num_stages, args.permutation_times)

    print("Finish!")
