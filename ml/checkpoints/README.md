# Checkpoints folder

This folder is where SwarKavach looks for a **trained** anti-spoofing
checkpoint to switch the anti-spoofing stage from
`prototype_demo` mode into `real_model` mode.

## What SwarKavach looks for

- `run1_best.pt`
- `run2_best.pt`

(same filenames the reference repository's own Streamlit demo,
`models/Anti_Spoofing/Model1/demo/app.py`, looks for.)

## Why this folder is empty by default

The reference repository
(`Dual-System-Framework-for-Neural-Voice-Cloning-and-Anti-Spoofing-Detection`)
does **not** commit any trained `.pt`/`.pth` checkpoint. Weights are
produced by running its own training pipeline. This was verified by
searching the entire cloned repository for `.pt`, `.pth`, and `.ckpt`
files - none exist.

## How to produce a real checkpoint (optional, not required for the demo)

From the reference repository:

```powershell
cd models\Anti_Spoofing\Model1
python run_all.py --setA_dir <path-to-setA> --setB_dir <path-to-setB>
```

This produces `models\run1_best.pt` inside that project. Copy it here
(`swar_kavach\ml\checkpoints\run1_best.pt`) and restart the inference
bridge. SwarKavach will then use the real WavLM + subband + AASIST
hybrid classifier for anti-spoofing, and the UI/API will report
`"anti_spoofing_mode": "real_model"` instead of `"prototype_demo"`.

## Speaker verification checkpoints

You do not need to place anything here for speaker verification. If the
`speechbrain` package is installed, its pretrained ECAPA-TDNN weights
are downloaded automatically on first use and cached under
`ml/checkpoints/ecapa_pretrained/`.
