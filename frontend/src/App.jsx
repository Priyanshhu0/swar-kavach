import { useEffect, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import StatusStrip from "./components/StatusStrip.jsx";
import VoiceInputPanel from "./components/VoiceInputPanel.jsx";
import ProcessingStages from "./components/ProcessingStages.jsx";
import WaveformSpectrogram from "./components/WaveformSpectrogram.jsx";
import SpoofResultCard from "./components/SpoofResultCard.jsx";
import SpeakerVerificationCard from "./components/SpeakerVerificationCard.jsx";
import RiskGauge from "./components/RiskGauge.jsx";
import RecommendationBanner from "./components/RecommendationBanner.jsx";
import AnalysisDetails from "./components/AnalysisDetails.jsx";
import { analyzeVoice, checkHealth } from "./lib/api.js";

const STAGE_COUNT = 5;
const STAGE_INTERVAL_MS = 550;

export default function App() {
  const [device, setDevice] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [stageIndex, setStageIndex] = useState(-1);
  const [completed, setCompleted] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lastAnalysis, setLastAnalysis] = useState(null);
  const stageTimer = useRef(null);

  useEffect(() => {
    checkHealth()
      .then((h) => setDevice(h.device))
      .catch(() => setDevice(null));
  }, []);

  function startStageAnimation() {
    let i = 0;
    setStageIndex(0);
    stageTimer.current = setInterval(() => {
      i += 1;
      if (i >= STAGE_COUNT - 1) {
        clearInterval(stageTimer.current);
        setStageIndex(STAGE_COUNT - 1);
      } else {
        setStageIndex(i);
      }
    }, STAGE_INTERVAL_MS);
  }

  function stopStageAnimation() {
    if (stageTimer.current) clearInterval(stageTimer.current);
  }

  async function handleAnalyze() {
    if (!voiceFile) return;
    setError(null);
    setCompleted(false);
    setResult(null);
    setAnalyzing(true);
    startStageAnimation();

    try {
      const [data] = await Promise.all([
        analyzeVoice(voiceFile, referenceFile),
        new Promise((resolve) => setTimeout(resolve, STAGE_INTERVAL_MS * (STAGE_COUNT - 1))),
      ]);
      stopStageAnimation();
      setResult(data);
      setCompleted(true);
      setLastAnalysis(
        new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      );
    } catch (err) {
      stopStageAnimation();
      setStageIndex(-1);
      setError(err.message || "Something went wrong while analyzing this file.");
    } finally {
      setAnalyzing(false);
    }
  }

  const modeLabel = result
    ? result.anti_spoofing.mode === "real_model"
      ? "Real model inference"
      : "Prototype / demo analysis"
    : null;

  return (
    <div className="min-h-screen">
      <Header device={device} />

      <main className="max-w-5xl mx-auto px-4 md:px-8 py-8 space-y-5">
        <StatusStrip
          device={device}
          mode={modeLabel}
          lastAnalysis={lastAnalysis}
          overallRisk={result?.risk}
        />

        <VoiceInputPanel
          voiceFile={voiceFile}
          setVoiceFile={setVoiceFile}
          referenceFile={referenceFile}
          setReferenceFile={setReferenceFile}
          onAnalyze={handleAnalyze}
          analyzing={analyzing}
          error={error}
        />

        <ProcessingStages activeIndex={stageIndex} completed={completed} />

        {completed && result && (
          <>
            <WaveformSpectrogram audio={result.audio} />
            <SpoofResultCard result={result.anti_spoofing} />
            <SpeakerVerificationCard result={result.speaker_verification} />
            <RiskGauge risk={result.risk} />
            <RecommendationBanner risk={result.risk} />
            <AnalysisDetails result={result} />
          </>
        )}
      </main>

      <footer className="max-w-5xl mx-auto px-4 md:px-8 pb-10 pt-2 text-[11px] text-faint">
        SwarKavach — prototype. no audio
        stores at Database.
      </footer>
    </div>
  );
}
