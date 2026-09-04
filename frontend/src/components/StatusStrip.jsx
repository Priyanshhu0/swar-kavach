import { Activity, Cpu, Clock, Gauge } from "lucide-react";

function StatCard({ icon: Icon, label, value, valueClass = "text-ink" }) {
  return (
    <div className="panel px-4 py-3.5 flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-panel2 border border-edge flex items-center justify-center shrink-0">
        <Icon size={16} className="text-muted" />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] text-faint leading-none mb-1">{label}</p>
        <p className={`text-sm font-medium truncate ${valueClass}`}>{value}</p>
      </div>
    </div>
  );
}

const RISK_COLOR = {
  LOW: "text-cyan",
  MEDIUM: "text-amber",
  HIGH: "text-coral",
};

export default function StatusStrip({ device, mode, lastAnalysis, overallRisk }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <StatCard icon={Activity} label="System status" value="Online · Ready" valueClass="text-cyan" />
      <StatCard
        icon={Cpu}
        label="Detection mode"
        value={mode || "Awaiting first analysis"}
      />
      <StatCard icon={Clock} label="Last analysis" value={lastAnalysis || "None yet"} />
      <StatCard
        icon={Gauge}
        label="Overall risk"
        value={overallRisk ? `${overallRisk.score}/100 · ${overallRisk.level}` : "—"}
        valueClass={overallRisk ? RISK_COLOR[overallRisk.level] : "text-muted"}
      />
    </div>
  );
}
