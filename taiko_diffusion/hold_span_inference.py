from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.data.hold_span_dataset import hold_span_features
from taiko_diffusion.train_hold_span import make_model


def load_hold_span_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["config"], float(checkpoint["duration_log_scale"])


@torch.inference_mode()
def predict_hold_spans(
    model: torch.nn.Module,
    audio: np.ndarray,
    condition: np.ndarray,
    legal_mask: np.ndarray,
    active_mask: np.ndarray,
    duration_log_scale: float,
    threshold: float = 0.8,
    device: torch.device | None = None,
) -> list[tuple[int, int, float]]:
    device = device or next(model.parameters()).device
    batch = {
        "audio": torch.from_numpy(audio).unsqueeze(0).to(device),
        "chart_context": torch.empty((1, 0, audio.shape[-1]), dtype=torch.float32, device=device),
        "legal_mask": torch.from_numpy(legal_mask.astype(np.float32)).unsqueeze(0).to(device),
        "active_mask": torch.from_numpy(active_mask.astype(np.float32)).unsqueeze(0).to(device),
        "condition": torch.from_numpy(condition.astype(np.float32)).unsqueeze(0).to(device),
    }
    output = model(hold_span_features(batch), batch["condition"])
    probabilities = torch.sigmoid(output["exist_logits"][0]).cpu().numpy()
    duration_logs = output["duration_log"][0].cpu().numpy()
    candidates = []
    for query_index in np.flatnonzero(probabilities >= threshold):
        start = int(round(float(output["start"][0, query_index].item()) * max(len(active_mask) - 1, 1)))
        legal_frames = np.flatnonzero((legal_mask > 0.5) & (active_mask > 0.5))
        if legal_frames.size == 0:
            continue
        start = int(legal_frames[np.argmin(np.abs(legal_frames - start))])
        duration = max(1, int(round(np.expm1(duration_logs[query_index] * duration_log_scale))))
        end = min(start + duration, int(active_mask.sum()) - 1)
        end = int(legal_frames[np.argmin(np.abs(legal_frames - end))])
        if end > start:
            candidates.append((start, end, float(probabilities[query_index])))
    accepted = []
    for candidate in sorted(candidates, key=lambda span: span[2], reverse=True):
        if all(candidate[1] < span[0] or candidate[0] > span[1] for span in accepted):
            accepted.append(candidate)
    return sorted(accepted)


def spans_to_hold_channels(length: int, spans: list[tuple[int, int, float]]) -> np.ndarray:
    holds = np.zeros((length, 3), dtype=np.float32)
    for start, end, _score in spans:
        if not 0 <= start < end < length:
            continue
        holds[start, 0] = 1.0
        holds[start + 1 : end, 1] = 1.0
        holds[end, 2] = 1.0
    return holds
