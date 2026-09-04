const BASE = "/api";

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function analyzeVoice(voiceFile, referenceFile) {
  const form = new FormData();
  form.append("voice_file", voiceFile);
  if (referenceFile) form.append("reference_file", referenceFile);

  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = "Analysis failed.";
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore parse error */
    }
    throw new Error(detail);
  }

  return res.json();
}

export function formatDuration(sec) {
  if (sec == null) return "—";
  return `${sec.toFixed(2)}s`;
}
