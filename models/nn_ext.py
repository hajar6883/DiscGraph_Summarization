import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch
from torcheval.metrics.functional import binary_f1_score


class NN_EXT(L.LightningModule):
    def __init__(
        self,
        in_channels,
        hidden_layers,
        hidden_channels,
        out_channels,
        dropout,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.in_layer = nn.Sequential(
            nn.Linear(in_channels - 20, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels + 20, hidden_channels + 20),
                    nn.LayerNorm(hidden_channels + 20),
                    nn.ReLU(),
                )
                for _ in range(hidden_layers)
            ]
        )

        self.out = nn.Linear(hidden_channels + 20, out_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sentence_features = x[:, :-20]
        graph_speaker_features = x[:, -20:]

        sentence_features = self.in_layer(sentence_features)

        x = torch.cat((sentence_features, graph_speaker_features), dim=1)

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout)
            x = layer(x)

        x = F.dropout(x, p=self.dropout)
        x = self.out(x)
        return self.sigmoid(x)

    def extract_last_layer(self, x):
        x = self.in_layer(x)

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout)
            x = layer(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = F.binary_cross_entropy(output, target)
        self.log(
            "training_loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = binary_f1_score(output.squeeze() >= 0.5, target.squeeze())
        self.log("val_f1", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
