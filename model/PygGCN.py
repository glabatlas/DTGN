import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class GCNEncoder(nn.Module):

    def __init__(self, hidden_list, activation=nn.Sigmoid()):
        super(GCNEncoder, self).__init__()
        self.is_train = True
        self.activation = activation
        self.layers = nn.ParameterList(
            [GCNConv(hidden_list[i], hidden_list[i + 1]) for i in range(len(hidden_list) - 1)])

    def forward(self, x, edges):
        for layer in self.layers[:-1]:
            x = layer(x, edges)
            x = self.activation(x)
        x = self.layers[-1](x, edges)
        return x


class GCNDecoder(nn.Module):
    def __init__(self, hidden_list, activation=nn.ReLU()):
        super(GCNDecoder, self).__init__()
        self.is_train = True
        self.activation = activation
        self.layers = nn.ParameterList(
            [nn.Linear(hidden_list[i], hidden_list[i + 1]) for i in range(len(hidden_list) - 1)])

    def forward(self, x, edges):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
