# @Author : CyIce
# @Time : 2024/6/25 14:27
import logging
import os
import os
import requests
import json
import torch

model = torch.load(f"./out/HCV/model.pth", map_location="cpu")
hidden_feat = model.encoder(train_feat, train_edges).permute(1, 0, 2).reshape((train_feat.size(1), -1))
file_operation.save_features(f'out/{args.dataset}/features/{run_id}', f'features.csv',
                             hidden_feat.detach().numpy())