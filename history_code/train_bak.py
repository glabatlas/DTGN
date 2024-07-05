import torch
from tqdm import tqdm

from history_code.utils_bak import file_operation
from history_code.utils_bak import loss_curve
from torch_geometric.utils import negative_sampling


def train(model, feat, edges, optim, device, name, epochs=10):
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
            model.encoder.is_train = False
            hidden_feat = model.encoder(feat, edges)
            test_roc, test_ap = model.test(hidden_feat, edges, negative_edges)

        if i % (epochs / 5) == 0:
            model.encoder.is_train = False
            decode_feat = model.feat_decoder(hidden_feat, edges).permute(1, 0, 2).reshape((feat.size(1), -1))
            hidden_feat = model.encoder(feat, edges).permute(1, 0, 2).reshape((feat.size(1), -1))

            if i == epochs:
                file_operation.save_features(f'out/{name}/features/{run_id}', f'features.csv',
                                             hidden_feat.cpu().detach().numpy())
                file_operation.save_features(f'out/{name}/features/{run_id}', f'de_feat.csv',
                                             decode_feat.cpu().detach().numpy())
            else:
                file_operation.save_features(f'out/{name}/features/{run_id}', f'features{i}.csv',
                                             hidden_feat.cpu().detach().numpy())
                file_operation.save_features(f'out/{name}/features/{run_id}', f'de_feat{i}.csv',
                                             decode_feat.cpu().detach().numpy())

        progress_bar.update(1)
        print_str = f"total loss: " + "{:.4f}".format(total_loss.item()) + "  feat loss: " + "{:.4f}".format(
            feat_loss.item()) + "  struct loss: " + "{:.4f}".format(
            recon_loss.item()) + "  AOC:" + "{:.4f}".format(test_roc) + "  AP:" + "{:.4f}".format(test_ap)
        progress_bar.set_description(str(run_id) + f" :{print_str}", refresh=True)

    loss_dict = {'Total loss': history_loss[0], "Feat loss": history_loss[1], "Recon loss": history_loss[2]}
    loss_curve(loss_dict, name=name, rate=1)
    loss_curve(loss_dict, name=name, rate=0.7)
