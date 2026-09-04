"""
Applied upgrades in this file:
  - Upgrade 4: Phase-aware top-k layer masking
  - Upgrade 5: SpecAugment on the adapter feature map
  - Upgrade 6: Subband branch integration with gated fusion
  - Post-review fix: gate now drives the fused representation
  - Post-review fix: exact phase-1 layer masking before softmax

WavLM Large frontend + AASIST HS-GAL backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from aasist_modules import GraphAttentionLayer, GraphPool, HtrgGraphAttentionLayer
from subband_branch import SubbandCNNBranch


class WavLMAASIST(nn.Module):
    model_name = "wavlm_aasist"

    def __init__(
        self,
        wavlm_name: str = "microsoft/wavlm-large",
        n_ssl_layers: int = 24,
        ssl_dim: int = 1024,
        adapter_dim: int = 64,
        gat_dims: list | None = None,
        pool_ratios: list | None = None,
        temperatures: list | None = None,
        n_spectral_nodes: int = 23,
        n_temporal_nodes: int = 29,
        freeze_encoder: bool = True,
        fine_tune_from_layer: int = 20,
        input_seconds: float = 6.0,
        time_mask_param: int = 30,
        freq_mask_param: int = 8,
        use_subband_branch: bool = True,
    ):
        super().__init__()

        if gat_dims is None:
            gat_dims = [64, 32]
        if pool_ratios is None:
            pool_ratios = [0.5, 0.7, 0.5, 0.5]
        if temperatures is None:
            temperatures = [2.0, 2.0, 100.0, 100.0]

        self.wavlm_name = wavlm_name
        self.n_ssl_layers = n_ssl_layers
        self.ssl_dim = ssl_dim
        self.adapter_dim = adapter_dim
        self.fine_tune_from_layer = fine_tune_from_layer
        self.input_seconds = float(input_seconds)
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.use_subband_branch = use_subband_branch
        self.current_phase = 1
        self._encoder_requires_grad = False

        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        self.wavlm.feature_extractor._freeze_parameters()
        self.layer_weights = nn.Parameter(torch.ones(n_ssl_layers))
        self.layer_mask = nn.Parameter(torch.ones(n_ssl_layers))
        self.register_buffer(
            "phase1_hard_mask",
            torch.tensor(
                [0] * 18 + [1] * 6,
                dtype=torch.float32,
            ),
        )
        self.layer_softmax = nn.Softmax(dim=0)
        self._last_layer_softmax = None

        self.adapter = nn.Sequential(
            nn.Linear(ssl_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
        )

        self.spectral_pool = nn.AdaptiveAvgPool1d(n_spectral_nodes)
        self.temporal_pool = nn.AdaptiveAvgPool1d(n_temporal_nodes)
        self.pos_S = nn.Parameter(torch.randn(1, n_spectral_nodes, adapter_dim))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(adapter_dim, gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(adapter_dim, gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.subband_branch = SubbandCNNBranch(sr=16000) if use_subband_branch else None
        self.gate = nn.Sequential(
            nn.Linear(224, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        self.gate_proj = nn.Linear(224, 160)
        self.drop = nn.Dropout(0.5, inplace=False)
        self.drop_way = nn.Dropout(0.2, inplace=False)
        self.out_layer = nn.Linear(160, 2)

        if freeze_encoder:
            self.freeze_encoder()

    def set_phase(self, phase: int):
        self.current_phase = int(phase)

    def freeze_encoder(self):
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self._encoder_requires_grad = False

    def unfreeze_top_layers(self, unfreeze_top_k: int | None = None):
        n_layers = len(self.wavlm.encoder.layers)
        if unfreeze_top_k is not None:
            self.fine_tune_from_layer = max(n_layers - int(unfreeze_top_k), 0)
        for idx, layer in enumerate(self.wavlm.encoder.layers):
            trainable = idx >= self.fine_tune_from_layer
            for param in layer.parameters():
                param.requires_grad = trainable
        self._encoder_requires_grad = True

    def enable_gradient_checkpointing(self):
        self.wavlm.gradient_checkpointing_enable()

    def get_optimizer_param_groups(
        self,
        lr_aasist: float,
        lr_encoder: float,
        llrd_factor: float = 0.85,
        weight_decay_aasist: float = 0.01,
        weight_decay_encoder: float = 0.0,
    ):
        param_groups = []

        subband_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad or name.startswith("wavlm."):
                continue
            if name.startswith("subband_branch."):
                subband_params.append(param)
            else:
                head_params.append(param)

        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": lr_aasist,
                    "weight_decay": weight_decay_aasist,
                }
            )
        if subband_params:
            param_groups.append(
                {
                    "params": subband_params,
                    "lr": lr_aasist,
                    "weight_decay": weight_decay_aasist,
                }
            )

        n_layers = len(self.wavlm.encoder.layers)
        for idx in range(n_layers - 1, -1, -1):
            layer = self.wavlm.encoder.layers[idx]
            trainable_params = [p for p in layer.parameters() if p.requires_grad]
            if trainable_params:
                depth_from_top = (n_layers - 1) - idx
                param_groups.append(
                    {
                        "params": trainable_params,
                        "lr": lr_encoder * (llrd_factor ** depth_from_top),
                        "weight_decay": weight_decay_encoder,
                    }
                )
        return param_groups

    def _compute_layer_weight_distribution(self) -> torch.Tensor:
        logits = self.layer_weights * self.layer_mask
        if self.current_phase == 1:
            logits = logits.masked_fill(self.phase1_hard_mask == 0, -1e9)
        weights = self.layer_softmax(logits)
        self._last_layer_softmax = weights.detach()
        return weights

    def _encode_ssl(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self._encoder_requires_grad):
            outputs = self.wavlm(x, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states[1:], dim=1)
        weights = self._compute_layer_weight_distribution()
        return (hidden_states * weights.view(1, -1, 1, 1)).sum(dim=1)

    def apply_spec_augment(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if not training:
            return x

        x = x.clone()
        batch_size, time_steps, feat_dim = x.shape
        for b in range(batch_size):
            if self.time_mask_param > 0 and time_steps > 1:
                width = int(torch.randint(0, min(self.time_mask_param, time_steps) + 1, (1,), device=x.device).item())
                if width > 0:
                    start = int(torch.randint(0, max(time_steps - width + 1, 1), (1,), device=x.device).item())
                    x[b, start:start + width, :] = 0.0

            if self.freq_mask_param > 0 and feat_dim > 1:
                width = int(torch.randint(0, min(self.freq_mask_param, feat_dim) + 1, (1,), device=x.device).item())
                if width > 0:
                    start = int(torch.randint(0, max(feat_dim - width + 1, 1), (1,), device=x.device).item())
                    x[b, :, start:start + width] = 0.0
        return x

    def _graph_readout(self, feat: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size(0)
        feat_t = feat.permute(0, 2, 1)

        e_S = self.spectral_pool(feat_t).permute(0, 2, 1) + self.pos_S
        e_T = self.temporal_pool(feat_t).permute(0, 2, 1)

        out_S = self.pool_S(self.GAT_layer_S(e_S))
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        master1 = self.master1.expand(batch_size, -1, -1)
        master2 = self.master2.expand(batch_size, -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        t_max, _ = torch.max(torch.abs(out_T), dim=1)
        t_avg = torch.mean(out_T, dim=1)
        s_max, _ = torch.max(torch.abs(out_S), dim=1)
        s_avg = torch.mean(out_S, dim=1)

        return torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)

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

        normalized_last_hidden = F.normalize(last_hidden_raw, dim=1)
        normalized_subband = F.normalize(subband_out, dim=1)
        gated_vector = torch.cat(
            [
                gate_weights[:, 0:1] * normalized_last_hidden,
                gate_weights[:, 1:2] * normalized_subband,
            ],
            dim=1,
        )

        fused = self.gate_proj(gated_vector)
        logits = self.out_layer(self.drop(fused))

        outputs = {
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
                    "gated_vector": gated_vector,
                    "layer_weight_distribution": self._last_layer_softmax,
                }
            )
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self.extract_features(x, return_parts=False)
        return outputs["fused"], outputs["logits"]

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def get_layer_weight_distribution(self) -> np.ndarray:
        with torch.no_grad():
            weights = self._compute_layer_weight_distribution().cpu().numpy()
        return weights

    def get_model_size_mb(self, path: str | None = None) -> float:
        if path is not None:
            import os
            return os.path.getsize(path) / 1e6
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        return total_bytes / 1e6
