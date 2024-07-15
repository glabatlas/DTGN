# @Author : CyIce
# @Time : 2024/6/25 10:41

import argparse


def args_parsing():
    parser = argparse.ArgumentParser(description="Parsing args on main.py.")
    parser.add_argument('--name', type=str, default="LR", help="Name of the model.")
    parser.add_argument('--exp_path', type=str, default="./data/LR/exp.csv", help="Path to the expression data.")
    parser.add_argument('--net_path', type=str, default="./data/LR/network.csv", help="Path to the network data.")
    parser.add_argument('--mean', type=float, default=0, help="The mean for filtering gene expression data.")
    parser.add_argument('--var', type=float, default=0, help="The variance for filtering gene expression data.")
    parser.add_argument('--norm_type', type=str, default="id", help="Type of normalization.")
    parser.add_argument('--permutation_times', type=int, default=2000, help="Permutation times.")
    parser.add_argument('--encoder_layer', type=str, default="16,8,2", help="Structure of the encoder layers.")
    parser.add_argument('--decoder_layer', type=str, default="2,8,16", help="Structure of the decoder layers.")
    parser.add_argument('--activation', type=str, default="relu", help="Activation function for the neural network.")
    parser.add_argument('--train', type=str, default="false", help="Whether to train the model.")
    parser.add_argument('--wd', type=float, default="5e-5", help="Weight decay for the optimizer.")
    parser.add_argument('--lr', type=float, default="1e-4", help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=30000, help="Number of epochs for training.")
    parser.add_argument('--device', type=str, default="cpu", help="Device to run the model on. Either 'cpu' or 'gpu'.")
    parser.add_argument('--threshold', type=float, default="0.01", help="Threshold for the model.")
    return parser
