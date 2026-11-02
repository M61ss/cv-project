import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4, max_len=256):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, SL, _ = x.shape     # (batch_size, sequence_length, d_model)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding[:, :SL+1]

        x = self.encoder(x)
        x = self.fc(self.norm(x[:, 0]))

