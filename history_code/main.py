import pandas as pd
import torch
import numpy as np
import random
import argparse
import os

from history_code.model import MyGAE
from history_code.model import GCNEncoder, GCNDecoder
from train import train
from history_code.utils_bak import file_operation, graph_operation
from factor_net import factor_grn, permutation_test


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def args_parsing():
    parser = argparse.ArgumentParser(description='Parsing args on main.py')
    parser.add_argument('-dataset', type=str, default='LR')
    parser.add_argument('-activation', type=str, default='relu')
    parser.add_argument('-train', type=str, default='false')
    parser.add_argument('-method', type=str, default='DTGN')
    parser.add_argument('-wd', type=float, default='5e-5')
    parser.add_argument('-lr', type=float, default='1e-4')
    parser.add_argument('-epoch', type=int, default=30000)
    parser.add_argument('-device', type=str, default="gpu")
    parser.add_argument('-stage', type=int, default=10)
    return parser


# 将表达数据重新编码
def one_hot_encode(feat, num_intervals, origin_val=False):
    max_value = torch.max(feat)
    min_value = torch.min(feat)
    stage_nums = feat.shape[0]
    n = feat.shape[1]
    interval_width = (max_value - min_value) / num_intervals
    one_hot_feat = torch.zeros((stage_nums, n, num_intervals))
    one_hot_pos = torch.zeros((stage_nums, n, 1), dtype=torch.long)
    for i in range(stage_nums):
        for j in range(n):
            interval_index = min(num_intervals - 1, int((feat[i, j, 0] - min_value) // interval_width))
            if origin_val:
                one_hot_feat[i, j, interval_index] = feat[i, j, 0]
                one_hot_pos[i, j, 0] = interval_index
            else:
                one_hot_feat[i, j, interval_index] = 1
                one_hot_pos[i, j, 0] = interval_index
    one_hot_pos = one_hot_pos.reshape((-1))
    return one_hot_feat, one_hot_pos.reshape((-1))


def train_pyg_gcn(feat, encode_list, decode_list, run_id):
    train_edges = torch.tensor(edges).T
    train_feat = feat.T.unsqueeze(-1)
    train_feat, one_hot_pos = one_hot_encode(train_feat, encode_list[0])

    if args.train == 'true':
        model = MyGAE(GCNEncoder(encode_list, activation=activation), GCNDecoder(decode_list, activation=activation))
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
        train(model, train_feat, train_edges, one_hot_pos, optim, device, dataset=args.dataset,
              recon_feat=feat.T.unsqueeze(-1), epochs=args.epoch,
              run_id=run_id)
        torch.save(model, f"out/{args.dataset}/model/{run_id}.model")
    elif args.train == 'false':
        model = torch.load(f"out/{args.dataset}/model/{run_id}.model", map_location=device)
        hidden_feat = model.encoder(train_feat, train_edges).permute(1, 0, 2).reshape((train_feat.size(1), -1))
        file_operation.save_features(f'out/{args.dataset}/features/{run_id}', f'features.csv',
                                     hidden_feat.detach().numpy())
    else:
        raise NotImplementedError


if __name__ == '__main__':

    main_parser = args_parsing()
    args = main_parser.parse_args()

    dataset = args.dataset
    activation = torch.nn.Sigmoid() if args.activation == 'sigmoid' else torch.nn.ReLU()
    device = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")
    stage = args.stage



    train_net_path = f"./data/{args.dataset}/training_net.csv"
    train_node_path = f"./data/{args.dataset}/training_node.csv"
    mapping_path = f"./data/{args.dataset}/mapping.csv"

    edges = pd.read_csv(train_net_path).values
    gene_list = {node for pair in edges for node in pair}
    edges = edges.tolist()
    features = torch.tensor(pd.read_csv(train_node_path).iloc[:, 2:].values.tolist())
    file_operation.save_features(f'out/{args.dataset}/features/SSN', f'features.csv', features)
    nums_node = len(gene_list)
    g = graph_operation.create_network(edges, nums_node, is_bidirected=True, add_loop=False)
    is_connected, c = graph_operation.is_conneted_graph(edges)

    print(
        f"dataset:{args.dataset}, method:{args.method}, num_stage:{args.stage}, train:{args.train},device:{args.device},epoch:{args.epoch},lr:{args.lr}")
    print(f'图G是否连通: {is_connected}')
    print(f"网络信息：{g}")

    pvalue = 0.05
    if dataset == "LR":
        hidden = [16, 8, 2]
    elif dataset == "MI":
        hidden = [32, 8, 2]
    elif dataset == "HCV":
        hiddent = [24, 8, 2]
    else:
        raise NotImplementedError
    a = 0
    seed_everything(42)
    torch.cuda.empty_cache()

    train_pyg_gcn(features, hidden, hidden[::-1], "DTGN")
    factor_grn.get_factor_grn(dataset=dataset, stage=stage, run_id=args.method, threshold=pvalue,
                              mapping_path=mapping_path, edges_path=train_net_path)
    if args.method == 'DTGN':
        permutation_test.diff_exp_test(dataset=dataset, stage=stage, run_id=args.method, using_exp=False,
                                       global_net=False,
                                       pvalue=pvalue, times=100, mapping_path=mapping_path, type="tf")
    elif args.method == 'SSN':
        permutation_test.diff_exp_test(dataset=dataset, stage=stage, run_id=args.method, using_exp=True,
                                       global_net=False, times=100,
                                       mapping_path=mapping_path)
    else:
        raise NotImplementedError
