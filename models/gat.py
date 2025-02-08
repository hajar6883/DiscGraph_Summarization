import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import lightning as L


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        dropout,
    ):
        super().__init__()

        self.in_hidden = hidden_channels
        self.out_hidden = hidden_channels // heads
        self.droput = dropout

        self.conv_in = GATv2Conv(in_channels, self.out_hidden, heads)

        self.conv_hidden_1 = GATv2Conv(self.in_hidden, self.out_hidden, heads)

        self.conv_out = GATv2Conv(self.in_hidden, self.out_hidden, heads)

        self.linear = nn.Linear(self.in_hidden, out_channels)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.1)
        x = self.conv_in(x, edge_index)
        x = nn.BatchNorm1d(self.in_hidden)(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.25)
        x = self.conv_hidden_1(x, edge_index)
        x = nn.BatchNorm1d(self.in_hidden)(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.25)
        x = self.conv_out(x, edge_index)
        x = nn.BatchNorm1d(self.in_hidden)(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.25)
        x = self.linear(x)
        return x
