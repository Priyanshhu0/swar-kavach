# Demo data folder

No audio files are included in this project (no third-party audio can
legally be redistributed here). Place your own test clips like this:

```
data/
  genuine/     -> real human speech recordings (.wav/.mp3/.flac)
  spoof/       -> AI-generated / cloned / replayed speech, if you have any
  reference/   -> "trusted voice" samples used for speaker verification
```

## Getting quick test audio

- Record a few seconds of your own voice on your phone or laptop mic and
  save it as .wav - this becomes a `genuine/` sample and also works as a
  `reference/` sample for speaker verification.
- If you have access to any TTS/voice-cloning tool (e.g. the reference
  repository's `models/Voice_Generation/XTTS` pipeline), generate a clip
  of a cloned voice and place it under `spoof/`.
- Any short (3-10 second) speech clip works for the "ANALYZE VOICE" demo
  flow, even a single `genuine/` sample - a reference sample is optional
  and only needed to demo the speaker-verification card.

## Validating your files

Run:

```powershell
python scripts\prepare_demo.py
```

This checks every file under `data/genuine`, `data/spoof`, and
`data/reference` for: valid audio decoding, minimum duration, and
non-silence - the same checks the inference bridge performs - so you
find problems before demoing to judges, not during.
