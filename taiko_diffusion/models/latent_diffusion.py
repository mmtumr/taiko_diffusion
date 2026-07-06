from __future__ import annotations

import torch
from torch import nn

from taiko_diffusion.models.chart_diffusion import (
    AudioContextEncoder,
    CrossAttention1D,
    ResBlock,
    SinusoidalEmbedding,
    group_count,
)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, scale: float = 1.0):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -10.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.scale = float(scale)

    def sample(self) -> torch.Tensor:
        return (self.mean + self.std * torch.randn_like(self.mean)) * self.scale

    def mode(self) -> torch.Tensor:
        return self.mean * self.scale

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.mean(self.mean.square() + self.var - 1.0 - self.logvar)


class PlainResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return x + h


class ChartAutoencoder1D(nn.Module):
    def __init__(
        self,
        chart_channels: int = 2,
        latent_channels: int = 16,
        base_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chart_channels = int(chart_channels)
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv1d(chart_channels, base_channels, kernel_size=7, padding=3),
            nn.GroupNorm(group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels * 4, latent_channels, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(base_channels, chart_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, chart: torch.Tensor) -> torch.Tensor:
        return self.encoder(chart)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, chart: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(chart)
        logits = self.decode(latent)
        return logits, latent


class ChartAutoencoderKL1D(nn.Module):
    def __init__(
        self,
        chart_channels: int = 2,
        latent_channels: int = 16,
        base_channels: int = 64,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.chart_channels = int(chart_channels)
        self.latent_channels = int(latent_channels)
        self.scale = float(scale)
        self.encoder = nn.Sequential(
            nn.Conv1d(chart_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count(base_channels), base_channels),
            nn.SiLU(),
            PlainResBlock1D(base_channels, dropout),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            PlainResBlock1D(base_channels * 2, dropout),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
            PlainResBlock1D(base_channels * 4, dropout),
            nn.Conv1d(base_channels * 4, latent_channels * 2, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
            PlainResBlock1D(base_channels * 4, dropout),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            PlainResBlock1D(base_channels * 2, dropout),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(group_count(base_channels), base_channels),
            nn.SiLU(),
            PlainResBlock1D(base_channels, dropout),
            nn.ConvTranspose1d(base_channels, chart_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, chart: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.encoder(chart), scale=self.scale)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent / self.scale)

    def forward(self, chart: torch.Tensor, sample_posterior: bool = True) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(chart)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latent), posterior


def encode_chart_latent(autoencoder: nn.Module, chart: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
    encoded = autoencoder.encode(chart)
    if hasattr(encoded, "sample") and hasattr(encoded, "mode"):
        return encoded.sample() if sample_posterior else encoded.mode()
    return encoded


class MugStyleAudioScaleEncoder1D(nn.Module):
    def __init__(
        self,
        audio_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channel_mults = list(channel_mults)
        widths = [base_channels * mult for mult in self.channel_mults]
        self.conv_in = nn.Conv1d(audio_channels, widths[0], kernel_size=3, padding=1)
        self.levels = nn.ModuleList()
        in_width = widths[0]
        for index, width in enumerate(widths):
            layers: list[nn.Module] = []
            if index > 0:
                layers.append(nn.Conv1d(in_width, in_width, kernel_size=4, stride=2, padding=1))
            for block_index in range(num_res_blocks):
                if block_index == 0 and in_width != width:
                    layers.append(nn.Conv1d(in_width, width, kernel_size=1))
                    in_width = width
                layers.append(ResBlock(width, base_channels * 4, dropout))
                in_width = width
            self.levels.append(nn.Sequential(*layers))

    def forward(self, audio: torch.Tensor) -> list[torch.Tensor]:
        h = self.conv_in(audio)
        outputs = []
        dummy_emb = torch.zeros((audio.shape[0], h.shape[1] * 4), device=audio.device, dtype=audio.dtype)
        for level in self.levels:
            for layer in level:
                if isinstance(layer, ResBlock):
                    if dummy_emb.shape[1] != layer.embed.in_features:
                        dummy_emb = torch.zeros((audio.shape[0], layer.embed.in_features), device=audio.device, dtype=audio.dtype)
                    h = layer(h, dummy_emb)
                else:
                    h = layer(h)
            outputs.append(h)
        return outputs


class LatentUNet1D(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        cond_dim: int,
        audio_channels: int,
        base_channels: int = 128,
        channel_mults: list[int] | None = None,
        dropout: float = 0.1,
        audio_context_dim: int = 128,
        audio_context_tokens: int = 128,
        audio_attention_heads: int = 4,
        audio_fusion: str = "token",
        audio_scale_blocks: int = 2,
    ):
        super().__init__()
        channel_mults = channel_mults or [1, 2, 2]
        widths = [base_channels * mult for mult in channel_mults]
        embed_dim = base_channels * 4
        self.latent_channels = int(latent_channels)
        self.audio_channels = int(audio_channels)
        self.audio_fusion = str(audio_fusion)
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
        self.audio_context = AudioContextEncoder(audio_channels, audio_context_dim, audio_context_tokens)
        self.audio_scale_encoder = (
            MugStyleAudioScaleEncoder1D(
                audio_channels,
                base_channels,
                [1, *[max(1, width // base_channels) for width in widths]],
                num_res_blocks=int(audio_scale_blocks),
                dropout=dropout,
            )
            if self.audio_fusion == "mug_scale"
            else None
        )
        self.audio_down_projs = nn.ModuleList()
        self.audio_up_projs = nn.ModuleList()
        if self.audio_scale_encoder is not None:
            scale_widths = [base_channels, *widths]
            for width, scale_width in zip(widths, scale_widths):
                self.audio_down_projs.append(nn.Conv1d(scale_width, width, kernel_size=1))
            for width, scale_width in zip(reversed(widths), reversed(scale_widths[1:])):
                self.audio_up_projs.append(nn.Conv1d(scale_width, width, kernel_size=1))
        self.audio_context_embed = nn.Sequential(
            nn.Linear(audio_context_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_conv = nn.Conv1d(latent_channels, widths[0], kernel_size=5, padding=2)
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, width in enumerate(widths):
            self.down_blocks.append(ResBlock(width, embed_dim, dropout))
            self.down_attn.append(CrossAttention1D(width, audio_context_dim, audio_attention_heads, dropout))
            if index + 1 < len(widths):
                self.downsamples.append(nn.Conv1d(width, widths[index + 1], kernel_size=4, stride=2, padding=1))

        self.mid = ResBlock(widths[-1], embed_dim, dropout)
        self.mid_attn = CrossAttention1D(widths[-1], audio_context_dim, audio_attention_heads, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in range(len(widths) - 1, -1, -1):
            width = widths[index]
            self.up_attn.append(CrossAttention1D(width, audio_context_dim, audio_attention_heads, dropout))
            self.up_blocks.append(ResBlock(width * 2, embed_dim, dropout))
            self.up_blocks.append(nn.Conv1d(width * 2, width, kernel_size=1))
            if index > 0:
                self.upsamples.append(nn.ConvTranspose1d(width, widths[index - 1], kernel_size=4, stride=2, padding=1))

        self.out = nn.Sequential(
            nn.GroupNorm(group_count(widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv1d(widths[0], latent_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        audio_context, audio_global = self.audio_context(audio)
        audio_scales = self.audio_scale_encoder(audio) if self.audio_scale_encoder is not None else None
        emb = self.time_embed(t) + self.cond_embed(condition) + self.audio_context_embed(audio_global)
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            if audio_scales is not None:
                audio_at_scale = audio_scales[index]
                if audio_at_scale.shape[-1] != h.shape[-1]:
                    audio_at_scale = torch.nn.functional.interpolate(
                        audio_at_scale,
                        size=h.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                h = h + self.audio_down_projs[index](audio_at_scale)
            h = block(h, emb)
            h = self.down_attn[index](h, audio_context)
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)

        h = self.mid(h, emb)
        h = self.mid_attn(h, audio_context)

        upsample_index = 0
        for index in range(0, len(self.up_blocks), 2):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            up_index = index // 2
            if audio_scales is not None:
                audio_at_scale = audio_scales[-(up_index + 1)]
                if audio_at_scale.shape[-1] != h.shape[-1]:
                    audio_at_scale = torch.nn.functional.interpolate(
                        audio_at_scale,
                        size=h.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                h = h + self.audio_up_projs[up_index](audio_at_scale)
            h = self.up_attn[up_index](h, audio_context)
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[index](h, emb)
            h = self.up_blocks[index + 1](h)
            if upsample_index < len(self.upsamples):
                h = self.upsamples[upsample_index](h)
                upsample_index += 1
        return self.out(h)
