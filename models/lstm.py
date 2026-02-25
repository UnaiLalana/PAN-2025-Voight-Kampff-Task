import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, text, lengths=None):
        embedded = self.embedding(text)

        if lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits