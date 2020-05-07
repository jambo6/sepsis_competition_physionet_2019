import torch
from torch import nn


class MLP(nn.Module):
    """ Simple multi-layer perception module. """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    """ Simple RNN. """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, nonlinearity='tanh', bias=True,
                 dropout=0):
        super(RNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.total_hidden_size = num_layers * hidden_channels

        self.rnn = nn.RNN(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)

        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, x):
        hidden = self.rnn(x)[0]
        return self.linear(hidden)


class GRU(nn.Module):
    """ Standard GRU. """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, bias=True, dropout=0):
        super(GRU, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.total_hidden_size = num_layers * hidden_channels

        self.gru = nn.GRU(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, x):
        hidden = self.gru(x)[0]
        return self.linear(hidden)
