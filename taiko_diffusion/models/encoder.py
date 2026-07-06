from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.Dropout(dropout),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = ConvBlock(channels, dropout)
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.block(x))


class AttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TaikoEncoderV0(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        conv_channels: int = 128,
        downsample_layers: int = 4,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3)
        self.down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(downsample_layers)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=transformer_heads,
            dim_feedforward=conv_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = AttentionPool(conv_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.stem(x)
        x = self.down(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = self.pool(x)
        return self.head(x)


class TaikoEncoderGroupedHeads(nn.Module):
    def __init__(
        self,
        input_channels: int,
        target_columns: list[str],
        head_groups: dict[str, list[str]],
        conv_channels: int = 128,
        downsample_layers: int = 4,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_columns = list(target_columns)
        self.head_groups = {name: list(labels) for name, labels in head_groups.items()}

        assigned = {label for labels in self.head_groups.values() for label in labels}
        missing = [label for label in self.target_columns if label not in assigned]
        if missing:
            self.head_groups["misc"] = missing

        unknown = sorted(assigned - set(self.target_columns))
        if unknown:
            raise ValueError(f"Grouped heads contain unknown targets: {unknown}")

        self.stem = nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3)
        self.down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(downsample_layers)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=transformer_heads,
            dim_feedforward=conv_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = AttentionPool(conv_channels)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(conv_channels),
                    nn.Linear(conv_channels, latent_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim, len(labels)),
                )
                for name, labels in self.head_groups.items()
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.stem(x)
        x = self.down(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        pooled = self.pool(x)

        by_label: dict[str, torch.Tensor] = {}
        for group_name, labels in self.head_groups.items():
            pred = self.heads[group_name](pooled)
            for index, label in enumerate(labels):
                by_label[label] = pred[:, index]
        return torch.stack([by_label[label] for label in self.target_columns], dim=1)


class TaikoEncoderSparseHeads(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        sparse_targets: list[str],
        conv_channels: int = 128,
        downsample_layers: int = 4,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sparse_targets = list(sparse_targets)
        self.stem = nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3)
        self.down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(downsample_layers)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=transformer_heads,
            dim_feedforward=conv_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = AttentionPool(conv_channels)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim),
        )
        self.sparse_head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, max(latent_dim // 2, len(sparse_targets))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, len(sparse_targets)), len(sparse_targets)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, C, T]
        x = self.stem(x)
        x = self.down(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        pooled = self.pool(x)
        return {
            "regression": self.regression_head(pooled),
            "sparse_logits": self.sparse_head(pooled),
        }


class TaikoEncoderBranchedSparse(nn.Module):
    def __init__(
        self,
        input_channels: int,
        target_columns: list[str],
        core_targets: list[str],
        sparse_targets: list[str],
        conv_channels: int = 128,
        shared_downsample_layers: int = 2,
        branch_downsample_layers: int = 2,
        core_transformer_layers: int = 2,
        event_transformer_layers: int = 1,
        transformer_heads: int = 8,
        latent_dim: int = 256,
        dropout: float = 0.1,
        class_targets: list[str] | None = None,
        class_dims: dict[str, int] | None = None,
        class_detach: bool = False,
        class_branch_downsample_layers: int = 0,
        class_transformer_layers: int = 1,
    ):
        super().__init__()
        self.target_columns = list(target_columns)
        self.core_targets = list(core_targets)
        self.sparse_targets = list(sparse_targets)
        self.class_targets = list(class_targets or [])
        self.class_dims = dict(class_dims or {})
        self.class_detach = class_detach
        self.class_branch_downsample_layers = int(class_branch_downsample_layers)

        assigned = set(self.core_targets) | set(self.sparse_targets)
        missing = [label for label in self.target_columns if label not in assigned]
        unknown = sorted(assigned - set(self.target_columns))
        if missing:
            raise ValueError(f"Branched sparse encoder missing targets: {missing}")
        if unknown:
            raise ValueError(f"Branched sparse encoder unknown targets: {unknown}")
        for label in self.class_targets:
            if label not in self.target_columns:
                raise ValueError(f"Classification target is not a target column: {label}")
            if label not in self.class_dims:
                raise ValueError(f"Missing class dim for classification target: {label}")

        self.stem = nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3)
        self.shared_down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(shared_downsample_layers)]
        )
        self.core_down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(branch_downsample_layers)]
        )
        self.event_down = nn.Sequential(
            *[DownsampleBlock(conv_channels, dropout) for _ in range(branch_downsample_layers)]
        )

        def make_transformer(layers: int) -> nn.TransformerEncoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_channels,
                nhead=transformer_heads,
                dim_feedforward=conv_channels * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.core_transformer = make_transformer(core_transformer_layers)
        self.event_transformer = make_transformer(event_transformer_layers)
        self.core_pool = AttentionPool(conv_channels)
        self.event_pool = AttentionPool(conv_channels)
        if self.class_targets and self.class_branch_downsample_layers > 0:
            self.class_down = nn.Sequential(
                *[
                    DownsampleBlock(conv_channels, dropout)
                    for _ in range(self.class_branch_downsample_layers)
                ]
            )
            self.class_transformer = make_transformer(class_transformer_layers)
            self.class_pool = AttentionPool(conv_channels)
        else:
            self.class_down = None
            self.class_transformer = None
            self.class_pool = None
        self.core_head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, len(self.core_targets)),
        )
        self.event_regression_head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, max(latent_dim // 2, len(self.sparse_targets))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, len(self.sparse_targets)), len(self.sparse_targets)),
        )
        self.sparse_head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, max(latent_dim // 2, len(self.sparse_targets))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, len(self.sparse_targets)), len(self.sparse_targets)),
        )
        self.class_heads = nn.ModuleDict(
            {
                label: nn.Sequential(
                    nn.LayerNorm(conv_channels),
                    nn.Linear(conv_channels, max(latent_dim // 2, int(self.class_dims[label]))),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(latent_dim // 2, int(self.class_dims[label])), int(self.class_dims[label])),
                )
                for label in self.class_targets
            }
        )

    def encode_branch(
        self,
        x: torch.Tensor,
        down: nn.Module,
        transformer: nn.Module,
        pool: nn.Module,
    ) -> torch.Tensor:
        x = down(x)
        x = x.transpose(1, 2)
        x = transformer(x)
        return pool(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, C, T]
        shared = self.shared_down(self.stem(x))
        core = self.encode_branch(shared, self.core_down, self.core_transformer, self.core_pool)
        event = self.encode_branch(shared, self.event_down, self.event_transformer, self.event_pool)

        by_label: dict[str, torch.Tensor] = {}
        core_pred = self.core_head(core)
        for index, label in enumerate(self.core_targets):
            by_label[label] = core_pred[:, index]

        event_pred = self.event_regression_head(event)
        for index, label in enumerate(self.sparse_targets):
            by_label[label] = event_pred[:, index]

        output = {
            "regression": torch.stack(
                [by_label[label] for label in self.target_columns], dim=1
            ),
            "sparse_logits": self.sparse_head(event),
        }
        if self.class_heads:
            if self.class_down is not None and self.class_transformer is not None and self.class_pool is not None:
                class_source = shared.detach() if self.class_detach else shared
                class_input = self.encode_branch(
                    class_source,
                    self.class_down,
                    self.class_transformer,
                    self.class_pool,
                )
            else:
                class_input = core.detach() if self.class_detach else core
            output["class_logits"] = {
                label: self.class_heads[label](class_input) for label in self.class_targets
            }
        return output
