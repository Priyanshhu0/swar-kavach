"""
Applied upgrades in this file:
  - Restored hybrid-model compatibility wrapper
  - Explicit 192-d hybrid fused representation for tests/training
  - Preserved WavLMModel import target for patch-based smoke tests

Backwards-compatible hybrid model name wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from wavlm_aasist import WavLMAASIST


class HybridWavLMSubbandAASIST(WavLMAASIST):
    model_name = "hybrid_wavlm_subband_aasist"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_dim = 192
        self.branch_a_proj = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, self.feature_dim),
            nn.GELU(),
        )
        self.branch_b_proj = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, self.feature_dim),
            nn.GELU(),
        )
        self.hybrid_out_layer = nn.Linear(self.feature_dim, 2)

    def extract_features(self, x: torch.Tensor, return_parts: bool = False) -> dict:
        ssl_feat = self._encode_ssl(x)
        adapted = self.adapter(ssl_feat)
        adapted = self.apply_spec_augment(adapted, training=self.training)

        last_hidden_raw = self._graph_readout(adapted)

        if self.subband_branch is not None:
            subband_out = self.subband_branch(x)
        else:
            subband_out = torch.zeros(x.size(0), 64, device=x.device, dtype=x.dtype)

        combined_raw = torch.cat([last_hidden_raw, subband_out], dim=1)
        gate_weights = self.gate(combined_raw)

        branch_a = self.branch_a_proj(last_hidden_raw)
        branch_b = self.branch_b_proj(subband_out)
        fused = (
            gate_weights[:, 0:1] * F.normalize(branch_a, dim=1)
            + gate_weights[:, 1:2] * F.normalize(branch_b, dim=1)
        )
        logits = self.hybrid_out_layer(self.drop(fused))

        outputs = {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "fused": fused,
            "logits": logits,
            "last_hidden_raw": last_hidden_raw,
            "subband": subband_out,
            "gate": gate_weights,
        }
        if return_parts:
            outputs.update(
                {
                    "ssl_feat": ssl_feat,
                    "adapter_feat": adapted,
                    "combined_raw": combined_raw,
                    "layer_weight_distribution": self._last_layer_softmax,
                }
            )
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self.extract_features(x, return_parts=False)
        return outputs["fused"], outputs["logits"]
