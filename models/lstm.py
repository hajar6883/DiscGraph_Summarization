import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch
from torcheval.metrics.functional import binary_f1_score


class LSTM(L.LightningModule):
    def __init__(
        self,
        in_channels,
        hidden_layers,
        hidden_channels,
        out_channels,
        dropout,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(in_channels, hidden_channels)

        self.in_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                )
                for _ in range(hidden_layers)
            ]
        )

        self.out = nn.Linear(hidden_channels, out_channels)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.in_layer(x)

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout)
            x = layer(x)

        x = F.dropout(x, p=self.dropout)
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx): #takes a batch of data, performs a forward pass, computes the binary cross-entropy loss, and logs the training loss.
        inputs, target = batch
        output = self.forward(inputs)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log(
            "training_loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = binary_f1_score(output.squeeze() >= 0.0, target.squeeze())
        self.log("val_f1", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
