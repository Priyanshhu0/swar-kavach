import { ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";

const LEVEL_CONFIG = {
  LOW: { icon: ShieldCheck, border: "border-cyan/30", bg: "bg-cyan/8", text: "text-cyan" },
  MEDIUM: { icon: ShieldAlert, border: "border-amber/30", bg: "bg-amber/8", text: "text-amber" },
  HIGH: { icon: ShieldX, border: "border-coral/30", bg: "bg-coral/8", text: "text-coral" },
};

export default function RecommendationBanner({ risk }) {
  if (!risk) return null;
  const cfg = LEVEL_CONFIG[risk.level] || LEVEL_CONFIG.LOW;
  const Icon = cfg.icon;

  return (
    <section className={`panel p-5 md:p-6 border ${cfg.border} ${cfg.bg}`}>
      <div className="flex items-start gap-4">
        <Icon size={26} className={`${cfg.text} shrink-0 mt-0.5`} />
        <div>
          <p className={`font-display text-sm font-semibold ${cfg.text} mb-1`}>
            Security recommendation
          </p>
          <p className="text-ink text-sm leading-relaxed">{risk.recommendation}</p>
        </div>
      </div>
    </section>
  );
}
