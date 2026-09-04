const BASE = (
  import.meta.env.VITE_API_URL ||
  "https://swar-kavach.onrender.com"
).replace(/\/$/, "");

export async function checkHealth() {
  const res = await fetch(`${BASE}/api/health`);

  if (!res.ok) {
    throw new Error("Health check failed");
  }

  return res.json();
}

export async function analyzeVoice(voiceFile, referenceFile = null) {
  if (!voiceFile) {
    throw new Error("Voice file is required.");
  }

  const form = new FormData();

  // These names MUST match FastAPI:
  // voice_file
  // reference_file
  form.append("voice_file", voiceFile);

  if (referenceFile) {
    form.append("reference_file", referenceFile);
  }

  const res = await fetch(`${BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });

  let body = null;

  try {
    body = await res.json();
  } catch {
    // Server didn't return JSON
  }

  if (!res.ok) {
    const detail =
      body?.detail ||
      `Analysis failed. Server returned ${res.status}.`;

    throw new Error(detail);
  }

  return body;
}

export function formatDuration(sec) {
  if (sec == null) return "—";

  return `${Number(sec).toFixed(2)}s`;
}
