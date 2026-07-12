from __future__ import annotations

import torch
from torch import nn

from taiko_diffusion.models.encoder import AttentionPool


class Branch(nn.Module):
    def __init__(self, inputs: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inputs, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
        )
        self.pool = AttentionPool(width)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.tokens(x))


def head(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, outputs))


class V2MultimodalLocalAttentionEncoder(nn.Module):
    """Predict main first, then condition target-specific continuous heads on it."""

    def __init__(self, chart_channels: int = 102, hand_channels: int = 24, audio_channels: int = 132):
        super().__init__()
        self.chart = Branch(chart_channels)
        self.hand = Branch(hand_channels)
        self.audio = Branch(audio_channels)
        self.audio_norm = nn.LayerNorm(64)
        self.chart_norm = nn.LayerNorm(64)
        self.local_audio_attention = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.cross_pool = AttentionPool(64)
        self.main_head = head(64 * 3, 1)
        self.body_head = head(64 * 2 + 1, 3)      # stamina, handspeed, burst
        self.complex_head = head(64 * 4 + 1, 1)   # pooled streams + aligned audio context
        self.rhythm_head = head(64 * 2 + 1, 1)    # chart + aligned audio; no hand shortcut

    @staticmethod
    def local_mask(length: int, radius: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        return (positions[:, None] - positions[None, :]).abs() > radius

    def forward(self, chart: torch.Tensor, hand: torch.Tensor, audio: torch.Tensor, teacher_main: torch.Tensor | None = None) -> torch.Tensor:
        chart_tokens = self.chart.tokens(chart)
        audio_tokens = self.audio.tokens(audio)
        chart_z, hand_z, audio_z = self.chart.pool(chart_tokens), self.hand(hand), self.audio.pool(audio_tokens)
        length = min(chart_tokens.shape[1], audio_tokens.shape[1])
        chart_local, audio_local = chart_tokens[:, :length], audio_tokens[:, :length]
        aligned, _ = self.local_audio_attention(
            self.chart_norm(chart_local), self.audio_norm(audio_local), self.audio_norm(audio_local),
            attn_mask=self.local_mask(length, 5, chart.device), need_weights=False,
        )
        aligned_z = self.cross_pool(aligned + chart_local)
        main = self.main_head(torch.cat([chart_z, hand_z, audio_z], dim=1))
        condition = teacher_main if teacher_main is not None else main.detach()
        body = self.body_head(torch.cat([chart_z, hand_z, condition], dim=1))
        complex_value = self.complex_head(torch.cat([chart_z, hand_z, audio_z, aligned_z, condition], dim=1))
        rhythm = self.rhythm_head(torch.cat([chart_z, aligned_z, condition], dim=1))
        return torch.cat([main, body, complex_value, rhythm], dim=1)


class V2MultimodalEncoder(nn.Module):
    """Stable pooled-audio baseline retained for checkpoint compatibility."""

    def __init__(self, chart_channels: int = 102, hand_channels: int = 24, audio_channels: int = 132):
        super().__init__()
        self.chart, self.hand, self.audio = Branch(chart_channels), Branch(hand_channels), Branch(audio_channels)
        self.main_head = head(64 * 3, 1)
        self.body_head = head(64 * 2 + 1, 3)
        self.complex_head = head(64 * 3 + 1, 1)
        self.rhythm_head = head(64 * 2 + 1, 1)

    def forward(self, chart: torch.Tensor, hand: torch.Tensor, audio: torch.Tensor, teacher_main: torch.Tensor | None = None) -> torch.Tensor:
        chart_z, hand_z, audio_z = self.chart(chart), self.hand(hand), self.audio(audio)
        main = self.main_head(torch.cat([chart_z, hand_z, audio_z], dim=1))
        condition = teacher_main if teacher_main is not None else main.detach()
        body = self.body_head(torch.cat([chart_z, hand_z, condition], dim=1))
        complex_value = self.complex_head(torch.cat([chart_z, hand_z, audio_z, condition], dim=1))
        rhythm = self.rhythm_head(torch.cat([chart_z, audio_z, condition], dim=1))
        return torch.cat([main, body, complex_value, rhythm], dim=1)


class V2TechniqueHeadEncoder(nn.Module):
    """Use only hand-technique tracks relevant to each target family."""

    def __init__(self, chart_channels: int = 102, hand_channels: int = 48, audio_channels: int = 132):
        super().__init__()
        if hand_channels != 48:
            raise ValueError("Technique-head encoder requires the 48-channel v2 hand cache")
        self.chart, self.audio = Branch(chart_channels), Branch(audio_channels)
        self.body_hand = Branch(hand_channels)
        self.complex_hand = Branch(hand_channels)
        self.main_head = head(64 * 2, 1)
        self.body_head = head(64 * 2 + 1, 3)
        self.complex_head = head(64 * 3 + 1, 1)
        self.rhythm_head = head(64 * 2 + 1, 1)

    @staticmethod
    def complex_tracks(hand: torch.Tensor) -> torch.Tensor:
        # 均衡、逆半均衡、半换、全换: first four methods x four
        # tracks, from both average- and max-pooled halves.
        # Include 硬抗 and 半分工: needing either can itself be evidence of
        # compound-pattern difficulty.
        return hand

    def forward(self, chart: torch.Tensor, hand: torch.Tensor, audio: torch.Tensor, teacher_main: torch.Tensor | None = None) -> torch.Tensor:
        chart_z, audio_z = self.chart(chart), self.audio(audio)
        body_hand_z = self.body_hand(hand)
        complex_hand_z = self.complex_hand(self.complex_tracks(hand))
        main = self.main_head(torch.cat([chart_z, audio_z], dim=1))
        condition = teacher_main if teacher_main is not None else main.detach()
        body = self.body_head(torch.cat([chart_z, body_hand_z, condition], dim=1))
        complex_value = self.complex_head(torch.cat([chart_z, complex_hand_z, audio_z, condition], dim=1))
        rhythm = self.rhythm_head(torch.cat([chart_z, audio_z, condition], dim=1))
        return torch.cat([main, body, complex_value, rhythm], dim=1)
