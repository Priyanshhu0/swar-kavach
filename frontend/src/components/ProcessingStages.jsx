import { Check, Loader2 } from "lucide-react";

const STAGES = [
  "Audio preprocessing",
  "WavLM feature extraction",
  "Anti-spoofing analysis",
  "Speaker verification",
  "Risk calculation",
];

export default function ProcessingStages({ activeIndex, completed }) {
  if (activeIndex < 0 && !completed) return null;

  return (
    <section className="panel p-5 md:p-6">
      <h2 className="font-display text-base font-semibold text-ink mb-4">AI analysis pipeline</h2>
      <ol className="space-y-2.5">
        {STAGES.map((label, i) => {
          const isDone = completed || i < activeIndex;
          const isActive = !completed && i === activeIndex;
          return (
            <li
              key={label}
              className={`flex items-center gap-3 text-sm rounded-lg px-3 py-2.5 border transition-colors ${
                isDone
                  ? "border-cyan/25 bg-cyan/5 text-ink"
                  : isActive
                  ? "border-blue/30 bg-blue/5 text-ink"
                  : "border-edge text-faint"
              }`}
            >
              <span className="w-5 h-5 shrink-0 flex items-center justify-center rounded-full">
                {isDone ? (
                  <Check size={15} className="text-cyan" />
                ) : isActive ? (
                  <Loader2 size={15} className="animate-spin text-blue" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-faint/50" />
                )}
              </span>
              {label}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
