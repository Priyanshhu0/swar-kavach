"""
ml/wavlm_features.py

Real WavLM self-supervised feature extraction.

WavLM's *pretrained encoder weights* (microsoft/wavlm-base-plus) are
publicly released by Microsoft on HuggingFace and are a legitimate,
freely-downloadable checkpoint - unlike the reference repository's
trained anti-spoofing *classifier head*, which is not committed to the
repository (no run1_best.pt / run2_best.pt is present anywhere in it).

So, unlike the anti-spoofing classification stage, WavLM feature
extraction here is REAL model inference whenever the model can be
downloaded/cached (first run needs internet access on the machine
running this backend). If it cannot be loaded (no internet, disk
space, etc.), this module honestly reports that and the pipeline
falls back to classical spectral features for the demo risk engine -
it never fabricates WavLM embeddings.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

_lock = threading.Lock()
_state = {"tried": False, "model": None, "processor": None, "device": None, "error": None}


@dataclass
class WavLMResult:
    available: bool
    model_name: str
    embedding_dim: int | None
    num_layers: int | None
    layer_mean_activation: list | None   # per-layer L2 norm, for the judge-facing chart
    pooled_embedding_summary: dict | None  # small, safe-to-serialize summary stats
    error: str | None


def _try_load(device: torch.device):
    if _state["tried"]:
        return
    with _lock:
        if _state["tried"]:
            return
        try:
            from transformers import WavLMModel, Wav2Vec2FeatureExtractor

            processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
            model = WavLMModel.from_pretrained(WAVLM_MODEL_NAME, output_hidden_states=True)
            model.eval()
            model.to(device)
            _state["model"] = model
            _state["processor"] = processor
            _state["device"] = device
        except Exception as exc:  # offline, no disk space, package missing, etc.
            _state["error"] = str(exc)
        finally:
            _state["tried"] = True


def get_wavlm_status() -> dict:
    """Non-loading status check (does not trigger a download)."""
    return {
        "attempted": _state["tried"],
        "available": _state["model"] is not None,
        "error": _state["error"],
    }


@torch.no_grad()
def extract_wavlm_features(y_16k: np.ndarray, device: torch.device) -> WavLMResult:
    """Run real WavLM inference if the pretrained encoder is available.

    Returns available=False (never fake numbers) if the model could not
    be loaded on this machine.
    """
    _try_load(device)
    model = _state["model"]
    processor = _state["processor"]

    if model is None:
        return WavLMResult(
            available=False,
            model_name=WAVLM_MODEL_NAME,
            embedding_dim=None,
            num_layers=None,
            layer_mean_activation=None,
            pooled_embedding_summary=None,
            error=_state["error"] or "WavLM model not loaded.",
        )

    inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(_state["device"])
    outputs = model(input_values)
    hidden_states = outputs.hidden_states  # tuple(num_layers+1) of [1, T, D]

    layer_norms = [float(h.norm(p=2, dim=-1).mean().item()) for h in hidden_states]
    pooled = hidden_states[-1].mean(dim=1).squeeze(0)  # [D]

    return WavLMResult(
        available=True,
        model_name=WAVLM_MODEL_NAME,
        embedding_dim=int(pooled.shape[-1]),
        num_layers=len(hidden_states),
        layer_mean_activation=[round(v, 4) for v in layer_norms],
        pooled_embedding_summary={
            "mean": round(float(pooled.mean().item()), 5),
            "std": round(float(pooled.std().item()), 5),
            "l2_norm": round(float(pooled.norm(p=2).item()), 4),
        },
        error=None,
    )
