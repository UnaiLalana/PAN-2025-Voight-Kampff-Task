import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        dropout: float = 0.3,
        pooling: str = "cls",
        freeze_encoder: bool = False,
    ):
        super(TransformerClassifier, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pooling == "cls":
            pooled_output = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled_output = summed / summed_mask
        else:
            raise ValueError("Pooling must be 'cls' or 'mean'")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits