import { Gauge } from "lucide-react";

const LEVEL_COLOR = {
  LOW: { stroke: "#35D0BA", text: "text-cyan", bg: "bg-cyan/8", border: "border-cyan/30" },
  MEDIUM: { stroke: "#F5B942", text: "text-amber", bg: "bg-amber/8", border: "border-amber/30" },
  HIGH: { stroke: "#FF5470", text: "text-coral", bg: "bg-coral/8", border: "border-coral/30" },
};

function CircularGauge({ score, level }) {
  const size = 168;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score / 100);
  const color = LEVEL_COLOR[level] || LEVEL_COLOR.LOW;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="#1C2A47"
          strokeWidth={stroke}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color.stroke}
          strokeWidth={stroke}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-4xl font-bold text-ink tabular">{score}</span>
        <span className="text-[11px] text-faint -mt-1">/ 100</span>
      </div>
    </div>
  );
}

function FactorRow({ label, value }) {
  if (value == null) return null;
  return (
    <div className="flex justify-between text-xs py-1.5 border-b border-edge last:border-0">
      <span className="text-muted">{label}</span>
      <span className="font-mono tabular text-ink">{value}%</span>
    </div>
  );
}

export default function RiskGauge({ risk }) {
  if (!risk) return null;
  const color = LEVEL_COLOR[risk.level] || LEVEL_COLOR.LOW;
  const inputs = risk.factors?.inputs || {};

  return (
    <section className="panel p-5 md:p-6">
      <div className="flex items-center gap-2 mb-5">
        <Gauge size={17} className="text-cyan" />
        <h2 className="font-display text-base font-semibold text-ink">Impersonation risk score</h2>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6 mb-5">
        <CircularGauge score={risk.score} level={risk.level} />
        <div className="flex-1 w-full">
          <span
            className={`inline-block font-display text-lg font-bold px-3 py-1 rounded-lg border ${color.bg} ${color.border} ${color.text} mb-3`}
          >
            {risk.level} RISK
          </span>
          <FactorRow label="AI spoof probability" value={inputs.spoof_probability_pct} />
          <FactorRow label="Speaker mismatch" value={inputs.speaker_mismatch_pct} />
          <FactorRow label="Detection confidence" value={inputs.detection_confidence_pct} />
        </div>
      </div>

      <p className="text-[11px] text-faint font-mono leading-relaxed border-t border-edge pt-3">
        {risk.formula_text}
      </p>
    </section>
  );
}
