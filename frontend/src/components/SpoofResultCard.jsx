import { Brain, ShieldAlert, ShieldCheck, Info } from "lucide-react";

function ProbBar({ label, value, colorClass }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-muted">{label}</span>
        <span className="font-mono tabular text-ink">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-panel3 overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass}`}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function SpoofResultCard({ result }) {
  if (!result) return null;
  const isSpoof = result.label === "spoof";
  const isReal = result.mode === "real_model";

  return (
    <section className="panel p-5 md:p-6">
      <div className="flex items-center gap-2 mb-4">
        <Brain size={17} className="text-cyan" />
        <h2 className="font-display text-base font-semibold text-ink">Anti-spoofing result</h2>
        <span
          className={`ml-auto text-[10px] font-mono px-2 py-1 rounded-full border ${
            isReal
              ? "border-cyan/30 text-cyan bg-cyan/5"
              : "border-amber/30 text-amber bg-amber/5"
          }`}
        >
          {isReal ? "REAL MODEL MODE" : "PROTOTYPE / DEMO ANALYSIS"}
        </span>
      </div>

      <div
        className={`flex items-center gap-4 rounded-xl px-5 py-5 mb-5 border ${
          isSpoof ? "border-coral/30 bg-coral/8" : "border-cyan/30 bg-cyan/8"
        }`}
      >
        {isSpoof ? (
          <ShieldAlert size={30} className="text-coral shrink-0" />
        ) : (
          <ShieldCheck size={30} className="text-cyan shrink-0" />
        )}
        <div>
          <p
            className={`font-display text-xl font-bold tracking-tight ${
              isSpoof ? "text-coral" : "text-cyan"
            }`}
          >
            {isSpoof ? "AI-GENERATED / SPOOF" : "REAL / BONAFIDE"}
          </p>
          <p className="text-xs text-muted mt-0.5">
            Model confidence: <span className="font-mono text-ink">{(result.confidence * 100).toFixed(1)}%</span>
          </p>
        </div>
      </div>

      <div className="space-y-3 mb-4">
        <ProbBar label="Spoof probability" value={result.spoof_probability} colorClass="bg-coral" />
        <ProbBar label="Bonafide probability" value={result.bonafide_probability} colorClass="bg-cyan" />
      </div>

      <div className="flex gap-2 text-xs text-muted bg-panel2 border border-edge rounded-lg px-3.5 py-3">
        <Info size={14} className="shrink-0 mt-0.5 text-faint" />
        <p>
          <span className="text-ink font-medium">{result.model_name}.</span> {result.notes}
        </p>
      </div>
    </section>
  );
}
