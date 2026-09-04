import { useState } from "react";
import { ChevronDown, FileCode2 } from "lucide-react";

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-4 py-2 border-b border-edge last:border-0 text-xs">
      <span className="text-muted">{label}</span>
      <span className="font-mono text-ink text-right">{value ?? "—"}</span>
    </div>
  );
}

export default function AnalysisDetails({ result }) {
  const [open, setOpen] = useState(false);
  if (!result) return null;

  const { audio, wavlm, anti_spoofing, speaker_verification, risk, timing, system } = result;

  return (
    <section className="panel overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 md:px-6 py-4"
      >
        <div className="flex items-center gap-2">
          <FileCode2 size={17} className="text-cyan" />
          <span className="font-display text-base font-semibold text-ink">
            Analysis details
          </span>
          <span className="text-[11px] text-faint">for judges / technical review</span>
        </div>
        <ChevronDown
          size={18}
          className={`text-muted transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="px-5 md:px-6 pb-6 grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-[11px] uppercase tracking-wide text-faint mb-2">Preprocessing</p>
            <Row label="Sample rate" value={`${audio.sample_rate} Hz`} />
            <Row label="Total duration" value={`${audio.duration_sec}s`} />
            <Row label="Detected speech" value={`${audio.detected_speech_sec}s`} />
            <Row label="Silence ratio" value={audio.silence_ratio} />
            <Row label="Compute device" value={system.device.toUpperCase()} />
          </div>

          <div>
            <p className="text-[11px] uppercase tracking-wide text-faint mb-2">WavLM feature extraction</p>
            <Row label="Status" value={wavlm.available ? "Real inference" : "Unavailable on this machine"} />
            <Row label="Model" value={wavlm.model_name} />
            <Row label="Embedding dim" value={wavlm.embedding_dim} />
            <Row label="Hidden layers" value={wavlm.num_layers} />
            {!wavlm.available && wavlm.error && (
              <p className="text-[11px] text-faint mt-2 leading-relaxed">{wavlm.error}</p>
            )}
          </div>

          <div>
            <p className="text-[11px] uppercase tracking-wide text-faint mb-2">Anti-spoofing</p>
            <Row label="Mode" value={anti_spoofing.mode} />
            <Row label="Model" value={anti_spoofing.model_name} />
            <Row label="Confidence" value={`${(anti_spoofing.confidence * 100).toFixed(1)}%`} />
            <Row label="Spoof probability" value={`${(anti_spoofing.spoof_probability * 100).toFixed(1)}%`} />
          </div>

          <div>
            <p className="text-[11px] uppercase tracking-wide text-faint mb-2">Speaker verification</p>
            <Row label="Mode" value={speaker_verification.mode} />
            <Row
              label="Similarity"
              value={
                speaker_verification.similarity_percent != null
                  ? `${speaker_verification.similarity_percent}%`
                  : "n/a"
              }
            />
            <Row label="Verdict" value={speaker_verification.verdict || "n/a"} />
          </div>

          <div className="md:col-span-2">
            <p className="text-[11px] uppercase tracking-wide text-faint mb-2">Risk calculation</p>
            <p className="text-xs font-mono text-muted mb-2">{risk.formula_text}</p>
            <Row label="Final score" value={`${risk.score} / 100 (${risk.level})`} />
            <Row label="Inference time" value={`${timing.inference_ms} ms`} />
          </div>
        </div>
      )}
    </section>
  );
}
