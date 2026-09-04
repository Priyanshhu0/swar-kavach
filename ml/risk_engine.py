"""
ml/risk_engine.py

Transparent, deterministic impersonation risk scoring.

Risk (0-100) is a fixed weighted combination of:
  - spoof_probability        (from the anti-spoofing stage)
  - speaker_mismatch         (1 - speaker similarity, if a reference was given)
  - detection_confidence     (how confident the anti-spoofing stage is)

No randomness is used anywhere in this calculation. The same audio +
same reference will always produce the same score, and every
contributing factor is returned to the caller so the UI can show its
exact arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass


# Fixed, documented weights. These sum to 1.0 when a speaker reference
# is available; when no reference is available, speaker_mismatch is
# dropped and the remaining weights are re-normalized proportionally.
WEIGHTS_WITH_SPEAKER = {
    "spoof_probability": 0.55,
    "speaker_mismatch": 0.25,
    "confidence": 0.20,
}
WEIGHTS_WITHOUT_SPEAKER = {
    "spoof_probability": 0.70,
    "confidence": 0.30,
}


@dataclass
class RiskResult:
    score: int             # 0-100
    level: str              # LOW / MEDIUM / HIGH
    recommendation: str
    factors: dict           # every contributing value + its weight + its contribution
    formula_text: str


def _level(score: int) -> str:
    if score <= 30:
        return "LOW"
    if score <= 70:
        return "MEDIUM"
    return "HIGH"


def _recommendation(level: str) -> str:
    return {
        "LOW": "Voice appears authentic. Normal verification recommended.",
        "MEDIUM": "Potential anomaly detected. Perform secondary verification.",
        "HIGH": "High impersonation risk detected. Do not approve sensitive actions. Verify through callback/MFA.",
    }[level]


def compute_risk(
    spoof_probability: float,
    confidence: float,
    speaker_similarity_percent: float | None,
) -> RiskResult:
    spoof_probability = max(0.0, min(1.0, spoof_probability))
    confidence = max(0.0, min(1.0, confidence))

    if speaker_similarity_percent is not None:
        speaker_mismatch = max(0.0, min(1.0, 1.0 - (speaker_similarity_percent / 100.0)))
        weights = WEIGHTS_WITH_SPEAKER
        contributions = {
            "spoof_probability": weights["spoof_probability"] * spoof_probability,
            "speaker_mismatch": weights["speaker_mismatch"] * speaker_mismatch,
            "confidence": weights["confidence"] * confidence,
        }
        formula_text = (
            f"Risk = {weights['spoof_probability']:.2f}×SpoofProbability + "
            f"{weights['speaker_mismatch']:.2f}×SpeakerMismatch + "
            f"{weights['confidence']:.2f}×DetectionConfidence, scaled to 0-100"
        )
    else:
        weights = WEIGHTS_WITHOUT_SPEAKER
        contributions = {
            "spoof_probability": weights["spoof_probability"] * spoof_probability,
            "confidence": weights["confidence"] * confidence,
        }
        formula_text = (
            f"Risk = {weights['spoof_probability']:.2f}×SpoofProbability + "
            f"{weights['confidence']:.2f}×DetectionConfidence, scaled to 0-100 "
            "(no reference sample was provided, so speaker mismatch is excluded and "
            "the remaining weights are used directly)"
        )

    raw = sum(contributions.values())  # 0-1
    score = int(round(raw * 100))
    score = max(0, min(100, score))
    level = _level(score)

    factors = {
        "inputs": {
            "spoof_probability_pct": round(spoof_probability * 100, 1),
            "speaker_mismatch_pct": (
                round((1.0 - speaker_similarity_percent / 100.0) * 100, 1)
                if speaker_similarity_percent is not None
                else None
            ),
            "detection_confidence_pct": round(confidence * 100, 1),
        },
        "weights": weights,
        "weighted_contributions_pct": {k: round(v * 100, 1) for k, v in contributions.items()},
        "final_score": score,
    }

    return RiskResult(
        score=score,
        level=level,
        recommendation=_recommendation(level),
        factors=factors,
        formula_text=formula_text,
    )
