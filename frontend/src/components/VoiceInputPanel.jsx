import { useRef, useState, useEffect } from "react";
import { Mic, Upload, UserCheck, Loader2, X } from "lucide-react";

function FileSlot({ label, hint, file, onChange, onClear, icon: Icon, accept }) {
  const inputRef = useRef(null);
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const objUrl = URL.createObjectURL(file);
    setUrl(objUrl);
    return () => URL.revokeObjectURL(objUrl);
  }, [file]);

  return (
    <div className="panel p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon size={15} className="text-muted" />
          <span className="text-sm font-medium text-ink">{label}</span>
        </div>
        {file && (
          <button
            onClick={onClear}
            className="text-faint hover:text-coral transition-colors"
            aria-label={`Clear ${label}`}
          >
            <X size={15} />
          </button>
        )}
      </div>

      {!file ? (
        <button
          onClick={() => inputRef.current?.click()}
          className="border border-dashed border-edge rounded-lg py-6 flex flex-col items-center gap-2 text-faint hover:text-cyan hover:border-cyan/40 transition-colors"
        >
          <Upload size={18} />
          <span className="text-xs">{hint}</span>
        </button>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-muted font-mono truncate">{file.name}</p>
          {url && <audio controls src={url} className="w-full h-9" />}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onChange(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

export default function VoiceInputPanel({
  voiceFile,
  setVoiceFile,
  referenceFile,
  setReferenceFile,
  onAnalyze,
  analyzing,
  error,
}) {
  return (
    <section className="panel p-5 md:p-6">
      <div className="flex items-center gap-2 mb-4">
        <Mic size={17} className="text-cyan" />
        <h2 className="font-display text-base font-semibold text-ink">Voice input</h2>
      </div>

      <div className="grid md:grid-cols-2 gap-3">
        <FileSlot
          label="Voice sample to analyze"
          hint="Upload WAV / MP3 / FLAC"
          file={voiceFile}
          onChange={setVoiceFile}
          onClear={() => setVoiceFile(null)}
          icon={Mic}
          accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
        />
        <FileSlot
          label="Trusted voice sample (optional)"
          hint="For speaker verification"
          file={referenceFile}
          onChange={setReferenceFile}
          onClear={() => setReferenceFile(null)}
          icon={UserCheck}
          accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
        />
      </div>

      {error && (
        <div className="mt-4 text-sm text-coral bg-coral/10 border border-coral/25 rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      <button
        onClick={onAnalyze}
        disabled={!voiceFile || analyzing}
        className="mt-5 w-full flex items-center justify-center gap-2 rounded-xl py-4 font-display font-semibold text-void bg-cyan hover:bg-cyan-soft disabled:bg-panel3 disabled:text-faint disabled:cursor-not-allowed transition-colors text-sm tracking-wide"
      >
        {analyzing ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Analyzing voice…
          </>
        ) : (
          <>ANALYZE VOICE</>
        )}
      </button>
    </section>
  );
}
