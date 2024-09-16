# @Author : CyIce
# @Time : 2024/6/24 17:32

import torch
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import pandas as pd
from utils.file_operation import save_df


def loss_curve(name, data: dict, rate=1.0):
    """
    drawing the loss curve.
    """
    plt.figure()
    COLOR_LIST = ['b', 'g', 'r', 'k', 'y']
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('loss')
    last_loss = []
    for i, label in enumerate(sorted(data.keys(), reverse=True)):
        loss = data[label]
        last_loss.append(str(round(loss[-1], 4)))
        x_train_loss = range(int((1 - rate) * len(loss)), len(loss))
        y_total_loss = loss[int((1 - rate) * len(loss)):]
        plt.plot(x_train_loss, y_total_loss, linewidth=1, color=COLOR_LIST[i], linestyle="solid", label=label)

    plt.legend()
    plt.title('Loss curve : ' + last_loss[0] + " = " + "+".join(last_loss[1:]))
    plt.savefig(f'./out/{name}/loss-{rate}.png')
    plt.close()


def train(name, model, genes, feat, edges, optim, device, epochs):
    """
    Trains the DTGN model.

    Parameters:
    name (str): The name of the dataset or experiment.
    model (torch.nn.Module): The model to be trained.
    genes (list of str): List of gene symbols.
    feat (Tensor): The feature matrix.
    edges (Tensor): The edge indices.
    optim (torch.optim.Optimizer): The optimizer for training.
    device (str): Device to run the model on ('cpu' or 'cuda').
    epochs (int): Number of training epochs.

    Returns:
    np.ndarray: The hidden features extracted by the model.

    Example:
    >>> hidden_features = train("experiment", model, gene_list, features, edge_list, optimizer, 'cuda', 100)
    """
    global hidden_feat
    model.train()
    model = model.to(device)
    feat = feat.to(device)
    edges = edges.to(device)

    history_loss = [[], [], [], []]
    progress_bar = tqdm(total=epochs, ncols=160)

    tf_nums = int(torch.max(edges[0, :])) + 1
    negative_edges = negative_sampling(edges, (tf_nums, feat.size(1)), force_undirected=True)
    test_roc, test_ap = model.test(feat, edges, negative_edges)

    for i in range(1, epochs + 1):
        optim.zero_grad()
        model.encoder.is_train = True
        hidden_feat = model.encoder(feat, edges)
        feat_loss = torch.tensor([0]).to(device)

        negative_edges = negative_sampling(edges, (tf_nums, feat.size(1)), force_undirected=True)
        recon_loss = model.recon_loss(hidden_feat, edges, negative_edges)
        total_loss = feat_loss + recon_loss
        total_loss.backward()
        optim.step()

        with torch.no_grad():
            history_loss[0].append(total_loss.item())
            history_loss[1].append(feat_loss.item())
            history_loss[2].append(recon_loss.item())

        if i % 50 == 0:
            hidden_feat = model.encoder(feat, edges)
            test_roc, test_ap = model.test(hidden_feat, edges, negative_edges)

        if i == epochs:
            decode_feat = model.feat_decoder(hidden_feat, edges).permute(1, 0, 2).reshape((feat.size(1), -1))
            hidden_feat = model.encoder(feat, edges).permute(1, 0, 2).reshape((feat.size(1), -1))

            hidden_df = pd.DataFrame(hidden_feat.cpu().detach().numpy())
            decode_df = pd.DataFrame(decode_feat.cpu().detach().numpy())
            hidden_df.insert(0, 'GeneSymbol', genes)
            decode_df.insert(0, 'GeneSymbol', genes)
            hidden_header = ['GeneSymbol'] + [f"feature_{i}" for i in range(hidden_df.shape[1] - 1)]
            decode_header = ['GeneSymbol'] + [f"feature_{i}" for i in range(decode_df.shape[1] - 1)]
            save_df(hidden_df, f"./out/{name}", "features.csv", header=hidden_header)
            save_df(decode_df, f"./out/{name}", "de_features.csv", header=decode_header)

        progress_bar.update(1)
        print_str = f"total loss: " + "{:.4f}".format(total_loss.item()) + "  feat loss: " + "{:.4f}".format(
            feat_loss.item()) + "  recon loss: " + "{:.4f}".format(
            recon_loss.item()) + "  AUC:" + "{:.4f}".format(test_roc) + "  AP:" + "{:.4f}".format(test_ap)
        progress_bar.set_description(str(name) + f" :{print_str}", refresh=True)

    loss_dict = {'Total loss': history_loss[0], "Feat loss": history_loss[1], "Recon loss": history_loss[2]}
    loss_curve(name, loss_dict, rate=1)
    loss_curve(name, loss_dict, rate=0.7)

    return hidden_feat.cpu().detach().numpy()
