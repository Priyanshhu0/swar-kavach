"""
scripts/prepare_demo.py

Validates the sample audio placed under data/genuine, data/spoof, and
data/reference before a live demo, so problems (corrupted files, silent
files, files that are too short) surface now instead of in front of
judges.

Usage (from the project root):

    python scripts\\prepare_demo.py            (Windows)
    python scripts/prepare_demo.py             (macOS/Linux)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

DATA_DIRS = ["genuine", "spoof", "reference"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def find_audio_files(root: str) -> list[str]:
    files = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in AUDIO_EXTS:
            files.append(path)
    return files


def main() -> int:
    from preprocessing import analyze_audio, AudioLoadError  # noqa: E402

    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_root = os.path.join(project_root, "data")

    total_checked = 0
    total_ok = 0
    problems = []

    print("=" * 60)
    print("SwarKavach demo data check")
    print("=" * 60)

    for category in DATA_DIRS:
        folder = os.path.join(data_root, category)
        if not os.path.isdir(folder):
            print(f"\n[{category}] folder missing: {folder}")
            continue

        files = find_audio_files(folder)
        print(f"\n[{category}] {len(files)} audio file(s) found")

        if not files:
            print(f"  (empty - see data/README.md for how to add samples)")
            continue

        for path in files:
            total_checked += 1
            name = os.path.basename(path)
            try:
                audio = analyze_audio(path, name)
                total_ok += 1
                print(
                    f"  OK   {name:35s} "
                    f"{audio.duration_sec:6.2f}s  sr={audio.sample_rate:6d}  "
                    f"speech={audio.detected_speech_sec:5.2f}s"
                )
            except AudioLoadError as exc:
                problems.append((path, str(exc)))
                print(f"  FAIL {name:35s} {exc}")
            except Exception as exc:
                problems.append((path, f"unexpected error: {exc}"))
                print(f"  FAIL {name:35s} unexpected error: {exc}")

    print("\n" + "=" * 60)
    print(f"Checked {total_checked} file(s), {total_ok} OK, {len(problems)} problem(s)")
    if problems:
        print("\nFiles that need attention before demoing:")
        for path, msg in problems:
            print(f"  - {path}: {msg}")
        print("=" * 60)
        return 1

    if total_checked == 0:
        print("\nNo audio files found yet. Add some under data/genuine, data/spoof,")
        print("or data/reference, then re-run this script. See data/README.md.")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
