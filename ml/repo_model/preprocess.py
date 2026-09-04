"""
Applied upgrades in this file:
  - Upgrade 1: Adaptive percentile VAD

Audio preprocessing pipeline for voice anti-spoofing.
"""

from __future__ import annotations

import wave

import numpy as np
import torch
import torchaudio

from utils import SEED, set_seed


set_seed(SEED)

TARGET_SR = 16000
MAX_LEN_SECONDS = 6.0
HYBRID_INPUT_SECONDS = 5.0
MAX_SAMPLES = int(TARGET_SR * MAX_LEN_SECONDS)
PRE_EMPHASIS_COEFF = 0.97
VAD_FRAME_MS = 20
VAD_MIN_KEEP_SECONDS = 0.5


def _load_wave_stdlib(filepath: str) -> tuple[torch.Tensor, int]:
    """Fallback WAV reader when torchaudio backends are unavailable."""
    with wave.open(filepath, "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    data = data.reshape(-1, channels).T
    return torch.from_numpy(data), sr


def vad_trim(
    waveform: torch.Tensor,
    sr: int,
    percentile: float = 10,
) -> torch.Tensor:
    """
    Trim leading and trailing low-energy frames using a clip-adaptive percentile.

    Args:
        waveform: 1D waveform tensor.
        sr: Sample rate.
        percentile: Energy percentile used as the adaptive threshold.

    Returns:
        Trimmed waveform, or the original waveform if trimming would over-shorten it.
    """
    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    frame_len = int(sr * (VAD_FRAME_MS / 1000.0))
    min_keep_samples = int(sr * VAD_MIN_KEEP_SECONDS)

    if frame_len <= 0 or waveform.numel() < frame_len:
        return waveform

    n_frames = waveform.numel() // frame_len
    if n_frames == 0:
        return waveform

    trimmed_view = waveform[: n_frames * frame_len].view(n_frames, frame_len)
    rms_energy = trimmed_view.pow(2).mean(dim=1).sqrt()
    threshold = torch.quantile(rms_energy, percentile / 100.0)
    active = rms_energy >= threshold

    if not active.any():
        return waveform

    active_idx = active.nonzero(as_tuple=False).squeeze(1)
    start = int(active_idx[0].item() * frame_len)
    end = int(min((active_idx[-1].item() + 1) * frame_len, waveform.numel()))
    candidate = waveform[start:end]

    if candidate.numel() < min_keep_samples:
        return waveform
    return candidate


def load_and_preprocess(
    filepath: str,
    max_len_seconds: float = MAX_LEN_SECONDS,
) -> torch.Tensor:
    """
    Full preprocessing pipeline.

    Args:
        filepath: Path to audio file.

    Returns:
        1D float tensor of shape [target_samples].
    """
    target_samples = int(TARGET_SR * max_len_seconds)
    try:
        waveform, sr = torchaudio.load(filepath)
    except (RuntimeError, OSError):
        waveform, sr = _load_wave_stdlib(filepath)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
        sr = TARGET_SR

    waveform = waveform.squeeze(0)

    waveform_pe = torch.zeros_like(waveform)
    waveform_pe[0] = waveform[0]
    waveform_pe[1:] = waveform[1:] - PRE_EMPHASIS_COEFF * waveform[:-1]
    waveform = waveform_pe

    waveform = vad_trim(waveform, sr)

    rms = waveform.pow(2).mean().sqrt()
    if rms > 1e-9:
        target_rms = 10 ** (-23.0 / 20.0)
        waveform = waveform * (target_rms / rms)

    if waveform.numel() > target_samples:
        waveform = waveform[:target_samples]
    elif waveform.numel() < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))

    return waveform.float()


def compute_rms_energy(waveform: torch.Tensor) -> float:
    """Compute RMS energy of a waveform."""
    return waveform.pow(2).mean().sqrt().item()
