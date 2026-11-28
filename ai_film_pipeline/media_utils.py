"""Shared FFmpeg, media-probing, and frame-quantization helpers."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from mutagen.mp3 import MP3

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FRAME_RE = re.compile(r"frame=\s*(\d+)")


def get_ffmpeg_exe() -> str:
    """Return the bundled FFmpeg executable path from imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def quantize_to_fps(seconds: float, fps: int) -> tuple[int, float]:
    """Snap a duration upward to an exact integer number of frames at ``fps``."""
    n_frames = max(1, math.ceil(seconds * fps))
    return n_frames, n_frames / float(fps)


def probe_mp3_duration(path: Path) -> float:
    """Measure exact MP3 duration in seconds via mutagen."""
    audio = MP3(str(path))
    return float(audio.info.length)


def probe_media_duration(path: Path) -> float:
    """Measure duration in seconds using FFmpeg header inspection."""
    exe = get_ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = _DURATION_RE.search(proc.stderr)
    if not match:
        raise RuntimeError(f"Could not parse duration for {path}:\n{proc.stderr}")
    hours, mins, secs = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return hours * 3600.0 + mins * 60.0 + secs


def count_video_frames(path: Path) -> int:
    """Count exact decoded video frames in a rendered clip via FFmpeg null muxer."""
    exe = get_ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    matches = _FRAME_RE.findall(proc.stderr)
    if not matches:
        raise RuntimeError(f"Could not determine frame count for {path}:\n{proc.stderr}")
    return int(matches[-1])


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg with standard overwrite and error checking."""
    exe = get_ffmpeg_exe()
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc
