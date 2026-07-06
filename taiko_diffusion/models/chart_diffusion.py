from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


def group_count(channels: int) -> int:
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, channels: int, embed_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.embed = nn.Linear(embed_dim, channels)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.embed(emb).unsqueeze(-1)
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return x + h


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    half = dim // 2
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    ).unsqueeze(0)
    embedding = torch.cat([torch.sin(positions * frequencies), torch.cos(positions * frequencies)], dim=1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class AudioContextEncoder(nn.Module):
    def __init__(self, audio_channels: int, context_dim: int, context_tokens: int):
        super().__init__()
        self.context_tokens = int(context_tokens)
        self.stem = nn.Sequential(
            nn.Conv1d(audio_channels, context_dim, kernel_size=7, padding=3),
            nn.GroupNorm(group_count(context_dim), context_dim),
            nn.SiLU(),
            nn.Conv1d(context_dim, context_dim, kernel_size=5, padding=2),
            nn.GroupNorm(group_count(context_dim), context_dim),
            nn.SiLU(),
            nn.Conv1d(context_dim, context_dim, kernel_size=3, padding=1),
        )
        self.out_norm = nn.LayerNorm(context_dim)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_tokens > 0 and audio.shape[-1] != self.context_tokens:
            audio = torch.nn.functional.interpolate(
                audio,
                size=self.context_tokens,
                mode="linear",
                align_corners=False,
            )
        context = self.stem(audio).transpose(1, 2)
        context = context + sinusoidal_positions(context.shape[1], context.shape[2], context.device).unsqueeze(0)
        context = self.out_norm(context)
        pooled = context.mean(dim=1)
        return context, pooled


class CrossAttention1D(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int, dropout: float):
        super().__init__()
        heads = max(1, min(int(heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = torch.nn.functional.silu(self.norm(x)).transpose(1, 2)
        attended, _ = self.attn(query, context, context, need_weights=False)
        attended = self.out(attended).transpose(1, 2)
        return x + attended


class ChartUNet1D(nn.Module):
    def __init__(
        self,
        chart_channels: int,
        cond_dim: int,
        base_channels: int = 64,
        channel_mults: list[int] | None = None,
        dropout: float = 0.1,
        audio_channels: int = 0,
        audio_multiscale: bool = False,
        audio_fusion: str | None = None,
        audio_context_dim: int = 128,
        audio_context_tokens: int = 128,
        audio_attention_heads: int = 4,
    ):
        super().__init__()
        channel_mults = channel_mults or [1, 2, 4]
        self.chart_channels = chart_channels
        self.audio_channels = int(audio_channels)
        self.audio_multiscale = bool(audio_multiscale)
        self.audio_fusion = str(audio_fusion or ("add" if self.audio_channels > 0 else "none"))
        if self.audio_fusion not in {"none", "add", "cross_attention", "add_cross_attention"}:
            raise ValueError(f"Unsupported audio_fusion: {self.audio_fusion}")
        self.use_audio_add = self.audio_fusion in {"add", "add_cross_attention"}
        self.use_audio_attention = self.audio_fusion in {"cross_attention", "add_cross_attention"}
        embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        widths = [base_channels * mult for mult in channel_mults]
        self.in_conv = nn.Conv1d(chart_channels, widths[0], kernel_size=7, padding=3)
        self.audio_conv = (
            nn.Sequential(
                nn.Conv1d(self.audio_channels, widths[0], kernel_size=7, padding=3),
                nn.GroupNorm(group_count(widths[0]), widths[0]),
                nn.SiLU(),
                nn.Conv1d(widths[0], widths[0], kernel_size=3, padding=1),
            )
            if self.audio_channels > 0 and self.use_audio_add
            else None
        )
        self.audio_context = (
            AudioContextEncoder(self.audio_channels, int(audio_context_dim), int(audio_context_tokens))
            if self.audio_channels > 0 and self.use_audio_attention
            else None
        )
        self.audio_context_embed = (
            nn.Sequential(
                nn.Linear(int(audio_context_dim), embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if self.audio_context is not None
            else None
        )
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.audio_down_projs = nn.ModuleList()
        self.audio_down_attn = nn.ModuleList()
        for index, width in enumerate(widths):
            self.down_blocks.append(ResBlock(width, embed_dim, dropout))
            if self.audio_channels > 0 and self.use_audio_add and self.audio_multiscale:
                self.audio_down_projs.append(
                    nn.Sequential(
                        nn.Conv1d(self.audio_channels, width, kernel_size=5, padding=2),
                        nn.GroupNorm(group_count(width), width),
                        nn.SiLU(),
                        nn.Conv1d(width, width, kernel_size=3, padding=1),
                    )
                )
            if self.audio_context is not None:
                self.audio_down_attn.append(
                    CrossAttention1D(width, int(audio_context_dim), int(audio_attention_heads), dropout)
                )
            if index + 1 < len(widths):
                self.downsamples.append(nn.Conv1d(width, widths[index + 1], kernel_size=4, stride=2, padding=1))

        self.mid = ResBlock(widths[-1], embed_dim, dropout)
        self.audio_mid_attn = (
            CrossAttention1D(widths[-1], int(audio_context_dim), int(audio_attention_heads), dropout)
            if self.audio_context is not None
            else None
        )
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.audio_up_projs = nn.ModuleList()
        self.audio_up_attn = nn.ModuleList()
        for index in range(len(widths) - 1, -1, -1):
            width = widths[index]
            self.up_blocks.append(ResBlock(width * 2, embed_dim, dropout))
            self.up_blocks.append(nn.Conv1d(width * 2, width, kernel_size=1))
            if self.audio_channels > 0 and self.use_audio_add and self.audio_multiscale:
                self.audio_up_projs.append(
                    nn.Sequential(
                        nn.Conv1d(self.audio_channels, width, kernel_size=5, padding=2),
                        nn.GroupNorm(group_count(width), width),
                        nn.SiLU(),
                        nn.Conv1d(width, width, kernel_size=3, padding=1),
                    )
                )
            if self.audio_context is not None:
                self.audio_up_attn.append(
                    CrossAttention1D(width, int(audio_context_dim), int(audio_attention_heads), dropout)
                )
            if index > 0:
                self.upsamples.append(nn.ConvTranspose1d(width, widths[index - 1], kernel_size=4, stride=2, padding=1))

        self.out = nn.Sequential(
            nn.GroupNorm(group_count(widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv1d(widths[0], chart_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_embed(t) + self.cond_embed(condition)
        audio_context: torch.Tensor | None = None
        h = self.in_conv(x)
        if self.audio_channels > 0 and audio is None:
            raise ValueError("audio-conditioned ChartUNet1D requires an audio tensor")
        if self.audio_context is not None:
            assert audio is not None
            audio_context, audio_global = self.audio_context(audio)
            assert self.audio_context_embed is not None
            emb = emb + self.audio_context_embed(audio_global)
        if self.audio_conv is not None:
            if audio is None:
                raise ValueError("audio-conditioned ChartUNet1D requires an audio tensor")
            if audio.shape[-1] != h.shape[-1]:
                audio = torch.nn.functional.interpolate(audio, size=h.shape[-1], mode="linear", align_corners=False)
            h = h + self.audio_conv(audio)
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            if self.use_audio_add and self.audio_multiscale and audio is not None:
                audio_at_scale = torch.nn.functional.interpolate(
                    audio,
                    size=h.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
                h = h + self.audio_down_projs[index](audio_at_scale)
            h = block(h, emb)
            if audio_context is not None:
                h = self.audio_down_attn[index](h, audio_context)
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)
        h = self.mid(h, emb)
        if audio_context is not None:
            assert self.audio_mid_attn is not None
            h = self.audio_mid_attn(h, audio_context)
        upsample_index = 0
        for index in range(0, len(self.up_blocks), 2):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            up_index = index // 2
            if self.use_audio_add and self.audio_multiscale and audio is not None:
                audio_at_scale = torch.nn.functional.interpolate(
                    audio,
                    size=h.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
                h = h + self.audio_up_projs[up_index](audio_at_scale)
            if audio_context is not None:
                h = self.audio_up_attn[up_index](h, audio_context)
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[index](h, emb)
            h = self.up_blocks[index + 1](h)
            if upsample_index < len(self.upsamples):
                h = self.upsamples[upsample_index](h)
                upsample_index += 1
        return self.out(h)
