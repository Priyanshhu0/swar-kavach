"""
Applied upgrades in this file:
  - Compatibility updates for the upgraded WavLM+AASIST model registry

Model registry and checkpoint-aware builders.
"""

from __future__ import annotations

from copy import deepcopy

from hybrid_wavlm_subband_aasist import HybridWavLMSubbandAASIST
from wavlm_aasist import WavLMAASIST


BASELINE_MODEL_NAME = "wavlm_aasist"
HYBRID_MODEL_NAME = "hybrid_wavlm_subband_aasist"
DEFAULT_MODEL_NAME = BASELINE_MODEL_NAME


DEFAULT_MODEL_CONFIGS = {
    BASELINE_MODEL_NAME: {
        "input_seconds": 6.0,
        "freeze_encoder": True,
        "fine_tune_from_layer": 20,
        "wavlm_name": "microsoft/wavlm-large",
        "time_mask_param": 30,
        "freq_mask_param": 8,
        "use_subband_branch": True,
    },
    HYBRID_MODEL_NAME: {
        "input_seconds": 6.0,
        "freeze_encoder": True,
        "fine_tune_from_layer": 20,
        "wavlm_name": "microsoft/wavlm-large",
        "time_mask_param": 30,
        "freq_mask_param": 8,
        "use_subband_branch": True,
    },
}


def merge_model_config(model_name: str, config: dict | None = None) -> dict:
    merged = deepcopy(DEFAULT_MODEL_CONFIGS[model_name])
    if config:
        merged.update(config)
    return merged


def build_model(model_name: str = DEFAULT_MODEL_NAME, config: dict | None = None):
    cfg = merge_model_config(model_name, config)
    kwargs = dict(
        wavlm_name=cfg["wavlm_name"],
        freeze_encoder=cfg["freeze_encoder"],
        fine_tune_from_layer=cfg["fine_tune_from_layer"],
        input_seconds=cfg["input_seconds"],
        time_mask_param=cfg["time_mask_param"],
        freq_mask_param=cfg["freq_mask_param"],
        use_subband_branch=cfg["use_subband_branch"],
    )
    if model_name == BASELINE_MODEL_NAME:
        return WavLMAASIST(**kwargs)
    if model_name == HYBRID_MODEL_NAME:
        return HybridWavLMSubbandAASIST(**kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def build_model_from_checkpoint(checkpoint: dict):
    model_name = checkpoint.get("model_name", BASELINE_MODEL_NAME)
    model_config = checkpoint.get("model_config", {})
    model = build_model(model_name=model_name, config=model_config)
    if hasattr(model, "set_phase"):
        model.set_phase(int(checkpoint.get("current_phase", 2)))
    return model, model_name, merge_model_config(model_name, model_config)


def get_model_input_seconds(model_name: str, config: dict | None = None) -> float:
    return float(merge_model_config(model_name, config)["input_seconds"])
