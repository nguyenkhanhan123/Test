"""End-to-end local runner for Steps 1, 3, and 6 (+ optional validation)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from step1_script import build_pilot_episode
from step3_tts import process_episode_audio
from step6_render_local import render_pilot_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Audio-First AI film pipeline")
    parser.add_argument("--skip-script", action="store_true", help="Keep existing script.json")
    parser.add_argument("--skip-tts", action="store_true", help="Reuse existing WAV/word timings")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    script_path = base_dir / "script.json"

    if not args.skip_script:
        ep = build_pilot_episode()
        script_path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        print(f"[Step 1] Wrote {script_path}")

    if not args.skip_tts:
        asyncio.run(process_episode_audio(base_dir, script_path))

    final_mp4 = render_pilot_video(base_dir, script_path)
    print(f"Pipeline finished -> {final_mp4}")


if __name__ == "__main__":
    main()
