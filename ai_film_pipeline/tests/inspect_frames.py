"""Extract mid-speech sample frames from pilot_30s.mp4 for visual QC inspection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from media_utils import run_ffmpeg
from schema import EpisodeScript


def extract_qc_contact_sheet(
    script_name: str = "script.json",
    video_name: str = "pilot_30s.mp4",
    out_subdir: str = "qc_frames",
) -> Path:
    root = Path(__file__).resolve().parents[1]
    ep = EpisodeScript.model_validate_json((root / script_name).read_text(encoding="utf-8"))
    video = root / "output" / video_name
    qc_dir = root / "output" / out_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    timestamps: list[tuple[str, float]] = []
    offset = 0.0
    for shot in ep.shots:
        dur = shot.duration_sec or 4.0
        # Pick midpoint during spoken line so subtitles & motion frame are visible
        mid = offset + shot.lead_in_sec + (shot.speech_duration_sec or dur * 0.5) * 0.55
        timestamps.append((shot.shot_id, mid))
        offset += dur

    frame_paths: list[tuple[str, float, Path]] = []
    for shot_id, ts in timestamps:
        out_png = qc_dir / f"{shot_id}_mid.png"
        run_ffmpeg(["-ss", f"{ts:.3f}", "-i", str(video), "-frames:v", "1", str(out_png)])
        frame_paths.append((shot_id, ts, out_png))

    # Build 2x3 contact sheet (1280x1080) for crisp single-image review
    tile_w, tile_h = 640, 360
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 3), (12, 14, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, (shot_id, ts, p) in enumerate(frame_paths):
        col, row = idx % 2, idx // 2
        with Image.open(p) as im:
            thumb = im.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (col * tile_w, row * tile_h))
        draw.rectangle([col * tile_w + 8, row * tile_h + 8, col * tile_w + 170, row * tile_h + 30], fill=(0, 0, 0))
        draw.text((col * tile_w + 14, row * tile_h + 13), f"{shot_id} @ {ts:.2f}s", fill=(255, 220, 120))

    sheet_path = qc_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"Saved QC contact sheet: {sheet_path}")
    return sheet_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--script", default="script.json")
    p.add_argument("--video", default="pilot_30s.mp4")
    p.add_argument("--out", default="qc_frames")
    a = p.parse_args()
    extract_qc_contact_sheet(a.script, a.video, a.out)
