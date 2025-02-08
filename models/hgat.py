import torch.nn as nn
import torch
from torch_geometric.nn import GATv2Conv


class HGAT(nn.Module):
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

        self.w2w_1 = GATv2Conv(
            in_channels, in_channels // heads, heads, dropout=dropout
        )
        self.ln_1 = nn.LayerNorm(in_channels)
        self.w2w_2 = GATv2Conv(
            in_channels, in_channels // heads, heads, dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(in_channels)

        self.word_to_sent = GATv2Conv(
            in_channels, self.out_hidden, heads, dropout=dropout
        )
        self.ln_3 = nn.LayerNorm(self.in_hidden)

        self.s2s_1 = GATv2Conv(
            self.in_hidden,
            self.out_hidden,
            heads,
            edge_dim=17,
            dropout=dropout,
        )
        self.ln_4 = nn.LayerNorm(self.in_hidden)
        self.s2s_2 = GATv2Conv(
            self.in_hidden,
            self.out_hidden,
            heads,
            edge_dim=17,
            dropout=dropout,
        )
        self.ln_5 = nn.LayerNorm(self.in_hidden)

        self.red_1 = GATv2Conv(
            self.in_hidden, self.out_hidden, heads, dropout=dropout
        )
        self.ln_6 = nn.LayerNorm(self.in_hidden)
        self.red_2 = GATv2Conv(
            self.in_hidden, self.out_hidden, heads, dropout=dropout
        )
        self.ln_7 = nn.LayerNorm(self.in_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(self.in_hidden, self.in_hidden // 4),
            nn.LayerNorm(self.in_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.in_hidden // 4, out_channels),
        )

    def forward(
        self,
        x_words,
        x_sent,
        w2w_index,
        w2s_index,
        s2s_index,
        s2s_type,
        s2s_sim,
    ):
        x_words = self.w2w_1(x_words, w2w_index)
        x_words = self.ln_1(x_words)
        x_words = nn.functional.relu(x_words)

        x_words = self.w2w_1(x_words, w2w_index)
        x_words = self.ln_2(x_words)
        x_words = nn.functional.relu(x_words)

        x = torch.cat((x_words, x_sent), dim=0)

        x = self.word_to_sent(x, w2s_index)
        x = self.ln_3(x)
        x = nn.functional.relu(x)

        x_sent = x[-len(x_sent) :]

        x_sent = self.s2s_1(x=x_sent, edge_index=s2s_index, edge_attr=s2s_type)
        x_sent = self.ln_4(x_sent)
        x_sent = nn.functional.relu(x_sent)

        x_sent = self.s2s_2(x=x_sent, edge_index=s2s_index, edge_attr=s2s_type)
        x_sent = self.ln_5(x_sent)
        x_sent = nn.functional.relu(x_sent)

        x_red = self.red_1(x=x_sent, edge_index=s2s_sim)
        x_red = self.ln_6(x_red)
        x_red = nn.functional.relu(x_red)

        x_red = self.red_2(x=x_red, edge_index=s2s_sim)
        x_red = self.ln_7(x_red)
        x_red = nn.functional.relu(x_red)

        return self.classifier(x_red)
