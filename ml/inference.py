"""
ml/inference.py

SwarKavach local inference bridge.

This is the "smallest possible local Python inference bridge" the brief
asks for: a single FastAPI process that exposes one endpoint the React
frontend calls. There is no database, no auth, no cloud service - it is
a local process that loads models once and serves requests from
localhost.

Run with:
    uvicorn inference:app --port 8000 --reload
(see project README for exact Windows commands)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from preprocessing import analyze_audio, AudioLoadError
from wavlm_features import extract_wavlm_features, get_wavlm_status
from anti_spoofing import detect_spoof
from speaker_verification import compare_speakers
from risk_engine import compute_risk

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB, generous for a local demo

app = FastAPI(title="SwarKavach Inference Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only - not for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": _DEVICE.type,
        "device_name": (torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "CPU"),
        "wavlm": get_wavlm_status(),
        "anti_spoofing_checkpoint_present": os.path.isdir(
            os.path.join(os.path.dirname(__file__), "checkpoints")
        )
        and any(
            os.path.exists(os.path.join(os.path.dirname(__file__), "checkpoints", name))
            for name in ("run1_best.pt", "run2_best.pt")
        ),
    }


def _save_upload(upload: UploadFile, tmpdir: str) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext or 'unknown'}'. "
            f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    dest = os.path.join(tmpdir, f"upload{ext}")
    with open(dest, "wb") as f:
        content = upload.file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File too large for this local demo (25MB max).")
        f.write(content)
    return dest


@app.post("/api/analyze")
async def analyze(voice_file: UploadFile = File(...), reference_file: UploadFile | None = File(None)):
    start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="swarkavach_")
    try:
        voice_path = _save_upload(voice_file, tmpdir)

        try:
            audio = analyze_audio(voice_path, voice_file.filename or "voice_sample")
        except AudioLoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        ref_audio = None
        ref_path = None
        if reference_file is not None and reference_file.filename:
            ref_path = _save_upload(reference_file, tmpdir)
            try:
                ref_audio = analyze_audio(ref_path, reference_file.filename)
            except AudioLoadError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Trusted voice sample problem: {exc}"
                )

        # Stage: WavLM feature extraction (real if the pretrained encoder loads)
        wavlm_result = extract_wavlm_features(audio.y_mono_16k, _DEVICE)

        # Stage: anti-spoofing (real trained checkpoint if present, else honest heuristic)
        spoof_result = detect_spoof(voice_path, audio, _DEVICE)

        # Stage: speaker verification (optional)
        speaker_result = compare_speakers(
            audio.y_mono_16k, ref_audio.y_mono_16k if ref_audio is not None else None
        )

        # Stage: risk scoring (fully deterministic, transparent)
        risk_result = compute_risk(
            spoof_probability=spoof_result.spoof_probability,
            confidence=spoof_result.confidence,
            speaker_similarity_percent=(
                speaker_result.similarity_percent if speaker_result.available else None
            ),
        )

        elapsed_ms = round((time.time() - start) * 1000, 1)

        response = {
            "system": {
                "device": _DEVICE.type,
                "anti_spoofing_mode": spoof_result.mode,
                "wavlm_mode": "real" if wavlm_result.available else "unavailable",
                "speaker_mode": speaker_result.mode,
            },
            "audio": {
                "filename": audio.filename,
                "sample_rate": audio.sample_rate,
                "duration_sec": audio.duration_sec,
                "detected_speech_sec": audio.detected_speech_sec,
                "silence_ratio": audio.silence_ratio,
                "waveform_points": audio.waveform_points,
                "waveform_times": audio.waveform_times,
                "spectrogram_db": audio.spectrogram_db,
                "spectrogram_times": audio.spectrogram_times,
                "spectrogram_freqs": audio.spectrogram_freqs,
            },
            "wavlm": {
                "available": wavlm_result.available,
                "model_name": wavlm_result.model_name,
                "embedding_dim": wavlm_result.embedding_dim,
                "num_layers": wavlm_result.num_layers,
                "layer_mean_activation": wavlm_result.layer_mean_activation,
                "pooled_embedding_summary": wavlm_result.pooled_embedding_summary,
                "error": wavlm_result.error,
            },
            "anti_spoofing": {
                "mode": spoof_result.mode,
                "label": spoof_result.label,
                "spoof_probability": spoof_result.spoof_probability,
                "bonafide_probability": spoof_result.bonafide_probability,
                "confidence": spoof_result.confidence,
                "model_name": spoof_result.model_name,
                "notes": spoof_result.notes,
                "factor_breakdown": spoof_result.factor_breakdown,
            },
            "speaker_verification": {
                "available": speaker_result.available,
                "mode": speaker_result.mode,
                "similarity_percent": speaker_result.similarity_percent,
                "verdict": speaker_result.verdict,
                "model_name": speaker_result.model_name,
                "notes": speaker_result.notes,
            },
            "risk": {
                "score": risk_result.score,
                "level": risk_result.level,
                "recommendation": risk_result.recommendation,
                "factors": risk_result.factors,
                "formula_text": risk_result.formula_text,
            },
            "timing": {
                "inference_ms": elapsed_ms,
            },
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as exc:
        # Never leak a raw traceback to the UI; log it locally instead.
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed unexpectedly: {exc}. See server console for details.",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
