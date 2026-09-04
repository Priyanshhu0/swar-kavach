"""
Applied upgrades in this file:
  - Upgrade 6: Subband CNN branch
  - Post-review fix: GPU Butterworth-style filterbank via torch.fft
  - H100/CUDA optimization: no CPU-side filtering in forward

Raw-audio subband CNN branch for anti-spoofing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubbandCNNBranch(nn.Module):
    def __init__(self, sr: int = 16000, filter_order: int = 4):
        super().__init__()
        self.sr = sr
        self.filter_order = int(filter_order)
        self.register_buffer(
            "band_edges",
            torch.tensor(
                [
                    [100.0, 300.0],
                    [300.0, 1000.0],
                    [1000.0, 2000.0],
                    [2000.0, 4000.0],
                    [4000.0, 6000.0],
                    [6000.0, 7500.0],
                ],
                dtype=torch.float32,
            ),
        )
        self._filter_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        self.backbone = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=512, stride=160, padding=256, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def _build_filterbank(
        self,
        signal_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (signal_length, device, dtype)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        freqs = torch.fft.rfftfreq(signal_length, d=1.0 / self.sr).to(device=device, dtype=dtype)
        freqs_safe = freqs.clamp_min(1e-6)
        order = float(self.filter_order)

        filters = []
        for low, high in self.band_edges.to(device=device, dtype=dtype):
            high_pass = 1.0 / torch.sqrt(1.0 + (low / freqs_safe) ** (2.0 * order))
            high_pass = torch.where(freqs > 0, high_pass, torch.zeros_like(high_pass))
            low_pass = 1.0 / torch.sqrt(1.0 + (freqs_safe / high) ** (2.0 * order))
            band_pass = high_pass * low_pass
            filters.append(band_pass)

        filterbank = torch.stack(filters, dim=0)
        self._filter_cache[cache_key] = filterbank
        return filterbank

    def _apply_filterbank(self, waveform: torch.Tensor) -> torch.Tensor:
        signal_length = waveform.size(-1)
        spectrum = torch.fft.rfft(waveform.float(), dim=-1)
        filterbank = self._build_filterbank(signal_length, waveform.device, spectrum.real.dtype)
        filtered_spectrum = spectrum.unsqueeze(1) * filterbank.unsqueeze(0).to(dtype=spectrum.dtype)
        filtered = torch.fft.irfft(filtered_spectrum, n=signal_length, dim=-1)
        return filtered.to(dtype=waveform.dtype)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._apply_filterbank(waveform)
        x = self.backbone(x).squeeze(-1)
        return x
