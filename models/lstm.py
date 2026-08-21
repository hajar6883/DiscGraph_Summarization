import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM classifier over token ids. This is the model trained in
    notebooks/LSTM.ipynb, which gave the best F1 of the models we tried."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text):
        # text: (batch, seq_len) of vocab indices
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        x = hidden.view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)
