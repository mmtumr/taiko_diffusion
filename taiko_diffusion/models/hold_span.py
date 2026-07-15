from __future__ import annotations

import torch
from torch import nn

from taiko_diffusion.models.chart_diffusion import group_count


class HoldSpanHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        condition_dim: int,
        query_count: int = 64,
        hidden_dim: int = 192,
        context_tokens: int = 512,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        attention_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_count = int(query_count)
        self.context_tokens = int(context_tokens)
        widths = [64, 96, 128, hidden_dim]
        layers: list[nn.Module] = []
        current = int(input_channels)
        for width in widths:
            layers.extend(
                [
                    nn.Conv1d(current, width, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(group_count(width), width),
                    nn.SiLU(),
                ]
            )
            current = width
        self.input_encoder = nn.Sequential(*layers)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_position = nn.Parameter(torch.randn(context_tokens, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.queries = nn.Parameter(torch.randn(query_count, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.input_encoder(features)
        context = torch.nn.functional.adaptive_avg_pool1d(context, self.context_tokens).transpose(1, 2)
        condition_embedding = self.condition_embed(condition)
        context = context + self.context_position.unsqueeze(0) + condition_embedding.unsqueeze(1)
        context = self.context_encoder(context)
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1) + condition_embedding.unsqueeze(1)
        decoded = self.query_decoder(queries, context)
        output = self.output(decoded)
        start_logits = torch.einsum("bqh,bth->bqt", decoded, context) / float(decoded.shape[-1]) ** 0.5
        start_probability = torch.softmax(start_logits, dim=-1)
        positions = torch.linspace(0.0, 1.0, self.context_tokens, device=features.device, dtype=features.dtype)
        return {
            "exist_logits": output[..., 0],
            "start_logits": start_logits,
            "start": (start_probability * positions).sum(dim=-1),
            "duration_log": torch.sigmoid(output[..., 1]),
        }
