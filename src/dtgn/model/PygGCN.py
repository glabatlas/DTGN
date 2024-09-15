import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder.

    Parameters:
    hidden_list (list of int): A list specifying the number of units in each hidden layer.
    activation (callable, optional): The activation function to apply after each layer except the last. Defaults to nn.Sigmoid().

    Methods:
    forward(x, edges):
        Performs a forward pass through the GCN layers.

    Example:
    >>> encoder = GCNEncoder([64, 32, 16])
    >>> output = encoder(features, edge_index)
    """

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

    """
    Graph Convolutional Network (GCN) Decoder.

    Parameters:
    hidden_list (list of int): A list specifying the number of units in each hidden layer.
    activation (callable, optional): The activation function to apply after each layer except the last. Defaults to nn.ReLU().

    Methods:
    forward(x, edges):
        Performs a forward pass through the linear layers.

    Example:
    >>> decoder = GCNDecoder([16, 32, 64])
    >>> output = decoder(features, edge_index)
    """
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
