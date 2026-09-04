"""
Applied upgrades in this file:
  - Upgrade 7: Temperature scaling calibration
  - A100/CUDA throughput optimization
  - H100/CUDA throughput optimization

Post-training temperature calibration helpers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from utils import get_autocast_kwargs


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = float(max(temperature, 1e-6))
    return logits / temperature


@torch.no_grad()
def calibrate_temperature(model, val_loader, device) -> float:
    model.eval()
    all_logits = []
    all_labels = []

    for waveforms, labels, _ in val_loader:
        waveforms = waveforms.to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast(**get_autocast_kwargs(device)):
            outputs = model.extract_features(waveforms, return_parts=False)
            logits = outputs["logits"]
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    if not all_logits:
        return 1.0

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    def objective(temp: float) -> float:
        scaled = apply_temperature(logits, temp)
        return float(F.cross_entropy(scaled, labels).item())

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    if not result.success:
        return 1.0
    return float(result.x)
