"""
ml/anti_spoofing.py

Anti-spoofing (bonafide vs AI-generated/spoof) detection stage.

REAL MODEL MODE
----------------
If a trained checkpoint from the reference repository's Model1 project
(WavLM + subband + AASIST hybrid classifier) is placed at
`ml/checkpoints/run1_best.pt` (or run2_best.pt), this module loads it
using the *exact* architecture and loading logic from the reference
repository (vendored, unmodified, in ml/repo_model/), and runs real
inference: the trained classifier's calibrated spoof probability is
used directly.

The reference repository does not commit any trained checkpoint
(no .pt/.pth file exists anywhere in it - weights must be produced by
running the repo's own training pipeline, models/Anti_Spoofing/Model1
/run_all.py or src/train.py). So out of the box, this stage runs in:

PROTOTYPE / DEMO MODE
----------------------
A transparent, deterministic, non-random heuristic computed purely from
measurable acoustic properties of the uploaded audio (pitch stability,
spectral flatness, and spectral-centroid variance - properties known to
differ between natural speech and many neural vocoders). This is
explicitly NOT a trained ML classifier and is always labeled as such in
both the API response and the UI. It never uses randomness.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from preprocessing import AudioAnalysis

REPO_MODEL_DIR = os.path.join(os.path.dirname(__file__), "repo_model")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_CANDIDATES = ["run1_best.pt", "run2_best.pt"]

if REPO_MODEL_DIR not in sys.path:
    sys.path.insert(0, REPO_MODEL_DIR)


@dataclass
class SpoofResult:
    mode: str                # "real_model" or "prototype_demo"
    label: str                # "bonafide" or "spoof"
    spoof_probability: float  # 0-1
    bonafide_probability: float
    confidence: float
    model_name: str
    notes: str
    factor_breakdown: dict    # transparent contributing measurements


_real_model_cache = {"tried": False, "bundle": None, "error": None}


def _find_checkpoint() -> str | None:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    for name in CHECKPOINT_CANDIDATES:
        candidate = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_real_model(device: torch.device):
    if _real_model_cache["tried"]:
        return _real_model_cache["bundle"], _real_model_cache["error"]

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        _real_model_cache["tried"] = True
        _real_model_cache["error"] = (
            "No trained checkpoint found at ml/checkpoints/run1_best.pt or "
            "run2_best.pt. Train one with the reference repository's "
            "models/Anti_Spoofing/Model1/run_all.py, or place a compatible "
            "checkpoint there, to enable REAL MODEL MODE."
        )
        return None, _real_model_cache["error"]

    try:
        from model_factory import build_model_from_checkpoint
        from calibrate import apply_temperature

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_name, model_config = build_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state"])
        if hasattr(model, "set_phase"):
            model.set_phase(ckpt.get("current_phase", 2))
        model.to(device)
        model.eval()
        bundle = {
            "model": model,
            "model_name": model_name,
            "model_config": model_config,
            "threshold": ckpt["threshold"],
            "temperature": ckpt.get("temperature", 1.0),
            "checkpoint_path": ckpt_path,
        }
        _real_model_cache["bundle"] = bundle
        _real_model_cache["error"] = None
    except Exception as exc:
        _real_model_cache["bundle"] = None
        _real_model_cache["error"] = f"Failed to load checkpoint {ckpt_path}: {exc}"
    finally:
        _real_model_cache["tried"] = True

    return _real_model_cache["bundle"], _real_model_cache["error"]


@torch.no_grad()
def _run_real_model(bundle: dict, filepath: str, device: torch.device) -> SpoofResult:
    from preprocess import load_and_preprocess  # vendored, unmodified
    from calibrate import apply_temperature

    waveform = load_and_preprocess(
        filepath, max_len_seconds=float(bundle["model_config"].get("input_seconds", 6.0))
    ).unsqueeze(0).to(device)

    outputs = bundle["model"].extract_features(waveform, return_parts=True)
    logits = apply_temperature(outputs["logits"], bundle["temperature"])
    probs = torch.softmax(logits, dim=1)
    spoof_prob = float(probs[0, 1].item())
    bonafide_prob = float(probs[0, 0].item())
    label = "spoof" if spoof_prob >= bundle["threshold"] else "bonafide"

    return SpoofResult(
        mode="real_model",
        label=label,
        spoof_probability=round(spoof_prob, 4),
        bonafide_probability=round(bonafide_prob, 4),
        confidence=round(max(spoof_prob, bonafide_prob), 4),
        model_name=f"{bundle['model_name']} ({os.path.basename(bundle['checkpoint_path'])})",
        notes=(
            "Real inference using the trained WavLM + AASIST hybrid checkpoint "
            "from the reference repository, with temperature-scaled calibration "
            f"(threshold={bundle['threshold']:.3f}, temperature={bundle['temperature']:.3f})."
        ),
        factor_breakdown={
            "calibrated_threshold": round(float(bundle["threshold"]), 4),
            "temperature": round(float(bundle["temperature"]), 4),
        },
    )


def _heuristic_prototype(analysis: AudioAnalysis) -> SpoofResult:
    """Deterministic, transparent, non-ML fallback.

    Combines three measurable acoustic signals that the anti-spoofing
    literature associates (imperfectly - this is a coarse heuristic, not
    a trained detector) with synthetic/vocoded speech:

      - pitch_stability: neural TTS/vocoder output is often unnaturally
        smooth/periodic compared to natural pitch micro-variation.
      - spectral_flatness: some vocoders leave characteristic flatness/
        noise-floor artifacts in the spectrum.
      - low spectral variability across the clip: natural conversational
        speech usually has more dynamic spectral movement.
    """
    pitch_component = float(np.clip(analysis.pitch_stability, 0.0, 1.0))
    flatness_component = float(np.clip(analysis.spectral_flatness * 12.0, 0.0, 1.0))
    zcr_regularity = float(np.clip(1.0 - analysis.zero_crossing_rate * 8.0, 0.0, 1.0))

    weights = {"pitch_stability": 0.45, "spectral_flatness": 0.35, "zcr_regularity": 0.20}
    raw_score = (
        weights["pitch_stability"] * pitch_component
        + weights["spectral_flatness"] * flatness_component
        + weights["zcr_regularity"] * zcr_regularity
    )
    spoof_prob = float(np.clip(raw_score, 0.02, 0.98))
    bonafide_prob = 1.0 - spoof_prob
    label = "spoof" if spoof_prob >= 0.5 else "bonafide"

    return SpoofResult(
        mode="prototype_demo",
        label=label,
        spoof_probability=round(spoof_prob, 4),
        bonafide_probability=round(bonafide_prob, 4),
        confidence=round(max(spoof_prob, bonafide_prob), 4),
        model_name="Prototype / Demo Analysis (deterministic acoustic heuristic)",
        notes=(
            "No trained anti-spoofing checkpoint is present, so this result comes "
            "from a deterministic, non-ML heuristic over measured acoustic features. "
            "It is NOT a trained AASIST/WavLM prediction and should not be "
            "presented as one - it is a stand-in for the demo."
        ),
        factor_breakdown={
            "pitch_stability_component": round(pitch_component, 4),
            "spectral_flatness_component": round(flatness_component, 4),
            "zcr_regularity_component": round(zcr_regularity, 4),
            "weights": weights,
        },
    )


def detect_spoof(filepath: str, analysis: AudioAnalysis, device: torch.device) -> SpoofResult:
    bundle, error = _load_real_model(device)
    if bundle is not None:
        try:
            return _run_real_model(bundle, filepath, device)
        except Exception as exc:
            # Real model failed at inference time - fall back honestly rather
            # than crash the demo, but say exactly what happened.
            result = _heuristic_prototype(analysis)
            result.notes = (
                f"Real model checkpoint was found but inference failed ({exc}); "
                "falling back to the deterministic prototype heuristic. " + result.notes
            )
            return result
    result = _heuristic_prototype(analysis)
    if error:
        result.notes = result.notes + f" ({error})"
    return result
