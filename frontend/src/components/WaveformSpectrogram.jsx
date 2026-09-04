import { useEffect, useRef } from "react";
import { AudioWaveform } from "lucide-react";

function WaveformCanvas({ points }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !points?.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const mid = height / 2;
    const max = Math.max(0.001, ...points.map((v) => Math.abs(v)));
    const step = width / points.length;

    ctx.beginPath();
    ctx.strokeStyle = "#35D0BA";
    ctx.lineWidth = 1.3;
    points.forEach((v, i) => {
      const x = i * step;
      const y = mid - (v / max) * mid * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = "rgba(140,160,196,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();
  }, [points]);

  return <canvas ref={ref} className="w-full h-24 block" />;
}

function SpectrogramCanvas({ db, freqs }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !db?.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const freqBins = db.length;
    const timeBins = db[0].length;
    let min = Infinity;
    let max = -Infinity;
    for (const row of db) for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = Math.max(1, max - min);

    const cellW = width / timeBins;
    const cellH = height / freqBins;

    for (let f = 0; f < freqBins; f++) {
      for (let t = 0; t < timeBins; t++) {
        const norm = (db[f][t] - min) / range;
        // low-to-high energy: void -> blue -> cyan -> amber (readable + on-brand)
        const c = colorForIntensity(norm);
        ctx.fillStyle = c;
        // flip vertically: low freq at bottom
        const y = height - (f + 1) * cellH;
        ctx.fillRect(t * cellW, y, cellW + 0.6, cellH + 0.6);
      }
    }
  }, [db]);

  function colorForIntensity(n) {
    n = Math.max(0, Math.min(1, n));
    const stops = [
      [11, 18, 32],
      [58, 111, 209],
      [53, 208, 186],
      [245, 185, 66],
    ];
    const scaled = n * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const frac = scaled - i;
    const a = stops[i];
    const b = stops[i + 1];
    const r = Math.round(a[0] + (b[0] - a[0]) * frac);
    const g = Math.round(a[1] + (b[1] - a[1]) * frac);
    const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
    return `rgb(${r},${g},${bl})`;
  }

  return <canvas ref={ref} className="w-full h-32 block rounded-md" />;
}

export default function WaveformSpectrogram({ audio }) {
  if (!audio) return null;
  return (
    <section className="panel p-5 md:p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <AudioWaveform size={17} className="text-cyan" />
          <h2 className="font-display text-base font-semibold text-ink">Audio analysis</h2>
        </div>
        <div className="flex gap-4 text-xs font-mono text-muted tabular">
          <span>{audio.sample_rate} Hz</span>
          <span>{audio.duration_sec.toFixed(2)}s total</span>
          <span>{audio.detected_speech_sec.toFixed(2)}s speech</span>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <p className="text-[11px] text-faint mb-1.5">Waveform</p>
          <WaveformCanvas points={audio.waveform_points} />
        </div>
        <div>
          <p className="text-[11px] text-faint mb-1.5">Spectrogram</p>
          <SpectrogramCanvas db={audio.spectrogram_db} freqs={audio.spectrogram_freqs} />
        </div>
      </div>
    </section>
  );
}
