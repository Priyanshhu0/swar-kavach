import { Users, CheckCircle2, XCircle, Info } from "lucide-react";

export default function SpeakerVerificationCard({ result }) {
  if (!result) return null;

  if (!result.available) {
    return (
      <section className="panel p-5 md:p-6">
        <div className="flex items-center gap-2 mb-3">
          <Users size={17} className="text-cyan" />
          <h2 className="font-display text-base font-semibold text-ink">Speaker verification</h2>
        </div>
        <p className="text-sm text-muted">{result.notes}</p>
      </section>
    );
  }

  const isMatch = result.verdict === "MATCH";
  const isReal = result.mode === "real_model";

  return (
    <section className="panel p-5 md:p-6">
      <div className="flex items-center gap-2 mb-4">
        <Users size={17} className="text-cyan" />
        <h2 className="font-display text-base font-semibold text-ink">Speaker verification</h2>
        <span
          className={`ml-auto text-[10px] font-mono px-2 py-1 rounded-full border ${
            isReal
              ? "border-cyan/30 text-cyan bg-cyan/5"
              : "border-amber/30 text-amber bg-amber/5"
          }`}
        >
          {isReal ? "REAL MODEL MODE" : "PROTOTYPE SPEAKER VERIFICATION"}
        </span>
      </div>

      <div className="flex items-center justify-between gap-6 mb-4">
        <div className="flex items-center gap-3 text-sm text-muted">
          <span className="px-3 py-1.5 rounded-lg bg-panel2 border border-edge text-ink">
            Current voice
          </span>
          <span className="text-faint">vs</span>
          <span className="px-3 py-1.5 rounded-lg bg-panel2 border border-edge text-ink">
            Trusted voice
          </span>
        </div>
        <div className="text-right">
          <p className="text-2xl font-display font-bold text-ink tabular">
            {result.similarity_percent.toFixed(1)}%
          </p>
          <p className="text-[11px] text-faint -mt-1">similarity</p>
        </div>
      </div>

      <div
        className={`flex items-center gap-2 rounded-lg px-4 py-3 border font-display font-semibold text-sm ${
          isMatch ? "border-cyan/30 bg-cyan/8 text-cyan" : "border-coral/30 bg-coral/8 text-coral"
        }`}
      >
        {isMatch ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
        {result.verdict}
      </div>

      <div className="flex gap-2 text-xs text-muted bg-panel2 border border-edge rounded-lg px-3.5 py-3 mt-4">
        <Info size={14} className="shrink-0 mt-0.5 text-faint" />
        <p>
          <span className="text-ink font-medium">{result.model_name}.</span> {result.notes}
        </p>
      </div>
    </section>
  );
}
