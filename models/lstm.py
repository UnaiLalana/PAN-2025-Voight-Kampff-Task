import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        return [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.split()]

    def __getitem__(self, idx):
        tokens = self.encode(self.texts.iloc[idx])
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        }

def collate_lstm(batch):
    MAX_LEN = 512

    input_ids = []
    labels = []

    for item in batch:
        ids = item["input_ids"][:MAX_LEN]
        input_ids.append(ids)
        labels.append(item["labels"])

    padded = pad_sequence(input_ids, batch_first=True)

    return {
        "input_ids": padded,
        "labels": torch.stack(labels)
    }

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
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths=None):
        embedded = self.embedding(text).contiguous()
        embedded = self.embedding_dropout(embedded).contiguous()

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            _, (hidden, _) = self.lstm(packed)

        else:
            _, (hidden, _) = self.lstm(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)

        return self.fc(hidden)