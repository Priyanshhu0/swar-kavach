"""
ml/speaker_verification.py

Speaker verification stage: compares the analyzed voice against an
optional "trusted voice" reference sample.

REAL MODEL MODE
----------------
Reuses the same approach as the reference repository's
Audio_Embedding_Analysis/Compare_Model_Embeddings/model.py: SpeechBrain's
pretrained ECAPA-TDNN speaker-embedding model
(speechbrain/spkrec-ecapa-voxceleb), a real, publicly released speaker
verification embedding model, compared with cosine similarity. This
requires the `speechbrain` package and an internet connection on first
run to fetch the pretrained weights (they are then cached locally).

PROTOTYPE MODE
--------------
If SpeechBrain / the pretrained embedding model is unavailable, this
module falls back to a classical MFCC-based voiceprint (mean+std of
MFCCs, cosine similarity) and clearly labels the result
"Prototype Speaker Verification" - never presenting it as the trained
ECAPA-TDNN result.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import librosa

_lock = threading.Lock()
_state = {"tried": False, "classifier": None, "error": None}


@dataclass
class SpeakerResult:
    available: bool
    mode: str                # "real_model" or "prototype"
    similarity_percent: float | None
    verdict: str | None       # "MATCH" / "MISMATCH" / None
    model_name: str
    notes: str


def _try_load_ecapa():
    if _state["tried"]:
        return
    with _lock:
        if _state["tried"]:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="ml/checkpoints/ecapa_pretrained",
            )
            _state["classifier"] = classifier
        except Exception as exc:
            _state["error"] = str(exc)
        finally:
            _state["tried"] = True


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _mfcc_voiceprint(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1)])
    return vec


def compare_speakers(
    y_test_16k: np.ndarray, y_ref_16k: np.ndarray | None
) -> SpeakerResult:
    if y_ref_16k is None:
        return SpeakerResult(
            available=False,
            mode="none",
            similarity_percent=None,
            verdict=None,
            model_name="n/a",
            notes="No trusted voice sample was provided, so speaker verification was skipped.",
        )

    _try_load_ecapa()
    classifier = _state["classifier"]

    if classifier is not None:
        try:
            import torch

            with torch.no_grad():
                emb_test = classifier.encode_batch(torch.from_numpy(y_test_16k).float().unsqueeze(0))
                emb_ref = classifier.encode_batch(torch.from_numpy(y_ref_16k).float().unsqueeze(0))
            sim = _cosine(emb_test.squeeze().cpu().numpy(), emb_ref.squeeze().cpu().numpy())
            sim_pct = round(float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)) * 100, 1)
            verdict = "MATCH" if sim_pct >= 70.0 else "MISMATCH"
            return SpeakerResult(
                available=True,
                mode="real_model",
                similarity_percent=sim_pct,
                verdict=verdict,
                model_name="SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)",
                notes="Real speaker-embedding cosine similarity from the pretrained ECAPA-TDNN model.",
            )
        except Exception as exc:
            # Fall through to the prototype path but say why.
            fallback_note = f"ECAPA-TDNN inference failed ({exc}); using prototype fallback. "
    else:
        fallback_note = (
            f"SpeechBrain ECAPA-TDNN unavailable ({_state['error']}); using prototype fallback. "
        )

    vec_test = _mfcc_voiceprint(y_test_16k)
    vec_ref = _mfcc_voiceprint(y_ref_16k)
    sim = _cosine(vec_test, vec_ref)
    sim_pct = round(float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)) * 100, 1)
    verdict = "MATCH" if sim_pct >= 75.0 else "MISMATCH"
    return SpeakerResult(
        available=True,
        mode="prototype",
        similarity_percent=sim_pct,
        verdict=verdict,
        model_name="Prototype Speaker Verification (MFCC voiceprint, cosine similarity)",
        notes=(
            fallback_note
            + "This is a classical-features prototype, not a trained speaker-verification "
            "model - treat the similarity score as indicative only."
        ),
    )
