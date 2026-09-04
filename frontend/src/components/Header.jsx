import { ShieldCheck } from "lucide-react";

export default function Header({ device }) {
  return (
    <header className="flex items-center justify-between gap-4 px-6 md:px-10 py-6 border-b border-edge">
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-panel2 border border-edge">
          <ShieldCheck size={20} className="text-cyan" strokeWidth={2.2} />
        </div>
        <div>
          <h1 className="font-display text-xl md:text-2xl font-semibold tracking-tight text-ink">
            SwarKavach
          </h1>
          <p className="text-xs md:text-sm text-muted -mt-0.5">
            Real-Time Voice Integrity Protection
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs font-mono text-muted">
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            device ? "bg-cyan" : "bg-faint"
          }`}
        />
        <span>{device ? `Running on ${device.toUpperCase()}` : "Connecting..."}</span>
      </div>
    </header>
  );
}
