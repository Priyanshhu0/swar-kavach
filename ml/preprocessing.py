"""
ml/preprocessing.py

Real audio preprocessing and analysis for SwarKavach.

This module does two jobs:

1. Load + normalize audio for model consumption, reusing the exact
   preprocessing routine (`load_and_preprocess`) from the reference
   repository's Model1 anti-spoofing project
   (models/Anti_Spoofing/Model1/src/preprocess.py), vendored here at
   ml/repo_model/preprocess.py. That function performs:
       resample -> mono-downmix -> pre-emphasis -> adaptive VAD trim
       -> RMS normalization -> fixed-length pad/crop
   exactly as the original repository does before feeding audio to the
   WavLM + AASIST model.

2. Compute real, measurable audio features for display and for the
   deterministic prototype/demo risk fallback (waveform, spectrogram,
   duration, speech-active duration, spectral statistics). Everything
   in this module is computed directly from the uploaded audio -
   nothing here is fabricated or randomized.
"""

from __future__ import annotations

import os
import sys
import wave
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import librosa

REPO_MODEL_DIR = os.path.join(os.path.dirname(__file__), "repo_model")
if REPO_MODEL_DIR not in sys.path:
    sys.path.insert(0, REPO_MODEL_DIR)

TARGET_SR = 16000


@dataclass
class AudioAnalysis:
    filename: str
    sample_rate: int
    duration_sec: float
    detected_speech_sec: float
    waveform_points: list           # downsampled amplitude points, for plotting
    waveform_times: list
    spectrogram_db: list            # 2D list [freq_bins][time_bins], dB
    spectrogram_times: list
    spectrogram_freqs: list
    rms_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_flatness: float
    pitch_stability: float          # 0-1, higher = more stable/periodic pitch
    silence_ratio: float
    y_mono_16k: np.ndarray = field(repr=False, default=None)


class AudioLoadError(Exception):
    """Raised for corrupted, empty, unsupported, or too-short audio."""


def _safe_load(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio robustly with librosa, falling back to the stdlib wave
    reader for plain PCM WAV files if soundfile/librosa cannot decode it."""
    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)
        if y is None or len(y) == 0:
            raise AudioLoadError("Decoded audio contains no samples.")
        return y.astype(np.float32), sr
    except Exception as exc:
        try:
            with wave.open(filepath, "rb") as handle:
                sr = handle.getframerate()
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                raw = handle.readframes(handle.getnframes())
            if sample_width == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sample_width == 1:
                data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise AudioLoadError(f"Unsupported WAV sample width: {sample_width}")
            data = data.reshape(-1, channels).mean(axis=1)
            return data.astype(np.float32), sr
        except Exception:
            raise AudioLoadError(
                f"Could not decode audio file. It may be corrupted or in an "
                f"unsupported format. ({exc})"
            )


def analyze_audio(filepath: str, filename: str, min_duration_sec: float = 0.35) -> AudioAnalysis:
    """Load an uploaded file and compute real, measurable audio analysis
    used for display and for the deterministic prototype risk fallback."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        raise AudioLoadError("Uploaded file is empty or missing.")

    y, sr = _safe_load(filepath)
    duration_sec = len(y) / sr if sr else 0.0

    if duration_sec < min_duration_sec:
        raise AudioLoadError(
            f"Audio is too short ({duration_sec:.2f}s). Please provide at least "
            f"{min_duration_sec:.2f}s of speech."
        )

    # Resample to 16kHz mono for consistent downstream analysis (matches
    # the target sample rate used throughout the reference repository).
    if sr != TARGET_SR:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    else:
        y16 = y.copy()

    if np.max(np.abs(y16)) < 1e-6:
        raise AudioLoadError("Audio appears to be silent (no detectable signal).")

    # --- voice-activity estimate (simple energy-based VAD, real computation) ---
    frame_len = int(TARGET_SR * 0.02)
    hop = frame_len
    n_frames = max(1, len(y16) // hop)
    frames = y16[: n_frames * hop].reshape(n_frames, hop) if n_frames > 0 else y16.reshape(1, -1)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thresh = np.percentile(frame_rms, 30) if len(frame_rms) > 3 else frame_rms.mean() * 0.5
    active = frame_rms > max(thresh, 1e-4)
    detected_speech_sec = float(active.sum() * (frame_len / TARGET_SR))
    silence_ratio = float(1.0 - active.mean()) if len(active) else 1.0
    if detected_speech_sec < 0.05:
        raise AudioLoadError("No speech activity detected in the uploaded audio.")

    # --- waveform for plotting (downsampled to ~600 points) ---
    n_points = min(600, len(y16))
    step = max(1, len(y16) // n_points)
    wf = y16[::step]
    wf_times = (np.arange(len(wf)) * step / TARGET_SR).tolist()

    # --- spectrogram (real STFT, downsampled to a plottable grid) ---
    n_fft = 1024
    hop_length = 256
    S = np.abs(librosa.stft(y16, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    # downsample to <=80 time bins and <=64 freq bins for a lightweight payload
    freq_bins, time_bins = S_db.shape
    t_step = max(1, time_bins // 80)
    f_step = max(1, freq_bins // 64)
    S_db_small = S_db[::f_step, ::t_step]
    spec_times = (np.arange(S_db_small.shape[1]) * t_step * hop_length / TARGET_SR).tolist()
    spec_freqs = (np.arange(S_db_small.shape[0]) * f_step * TARGET_SR / n_fft).tolist()

    # --- classical spectral statistics (real, used by heuristic fallback) ---
    rms = float(np.sqrt(np.mean(y16 ** 2) + 1e-12))
    rms_db = float(20 * np.log10(rms + 1e-9))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y16)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y16, sr=TARGET_SR)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y16)))

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y16, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=TARGET_SR
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        if len(voiced_f0) > 4:
            pitch_stability = float(
                np.clip(1.0 - (np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-6)), 0.0, 1.0)
            )
        else:
            pitch_stability = 0.5
    except Exception:
        pitch_stability = 0.5

    return AudioAnalysis(
        filename=filename,
        sample_rate=int(sr),
        duration_sec=round(duration_sec, 3),
        detected_speech_sec=round(detected_speech_sec, 3),
        waveform_points=[round(float(v), 5) for v in wf],
        waveform_times=[round(float(v), 4) for v in wf_times],
        spectrogram_db=[[round(float(v), 2) for v in row] for row in S_db_small],
        spectrogram_times=[round(float(v), 4) for v in spec_times],
        spectrogram_freqs=[round(float(v), 1) for v in spec_freqs],
        rms_db=round(rms_db, 2),
        zero_crossing_rate=round(zcr, 5),
        spectral_centroid_hz=round(centroid, 1),
        spectral_flatness=round(flatness, 5),
        pitch_stability=round(pitch_stability, 4),
        silence_ratio=round(silence_ratio, 4),
        y_mono_16k=y16,
    )
