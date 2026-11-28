"""Procedural dark-fantasy cinematic BGM & SFX synthesizer (copyright-free WAV)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLE_RATE = 48_000


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([left, right], axis=1)
    stereo = np.clip(stereo, -0.98, 0.98)
    pcm = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _lowpass(signal: np.ndarray, cutoff_hz: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff_hz / (0.5 * SAMPLE_RATE), btype="low")
    return filtfilt(b, a, signal)


def _bandpass(signal: np.ndarray, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    b, a = butter(order, [low_hz / (0.5 * SAMPLE_RATE), high_hz / (0.5 * SAMPLE_RATE)], btype="band")
    return filtfilt(b, a, signal)


def generate_cinematic_bgm(out_path: Path, total_duration_sec: float) -> Path:
    """Synthesize a moody D-minor nocturnal cinematic pad + sub-bass + wind texture."""
    n = int(round(total_duration_sec * SAMPLE_RATE))
    t = np.linspace(0.0, total_duration_sec, n, endpoint=False)
    rng = np.random.default_rng(42)

    # Chord progression (D minor -> Bb major7 -> G minor9 -> A7sus4 -> D minor)
    chords = [
        (0.00, 0.25, [73.42, 146.83, 220.00, 261.63, 349.23]),  # Dm7
        (0.25, 0.50, [58.27, 116.54, 174.61, 220.00, 293.66]),  # Bbmaj7
        (0.50, 0.75, [49.00, 146.83, 196.00, 233.08, 329.63]),  # Gm9
        (0.75, 1.01, [55.00, 110.00, 164.81, 220.00, 293.66]),  # A7sus4
    ]

    pad_l = np.zeros(n, dtype=np.float64)
    pad_r = np.zeros(n, dtype=np.float64)

    for start_frac, end_frac, freqs in chords:
        i0 = max(0, int(start_frac * n))
        i1 = min(n, int(end_frac * n))
        seg_len = i1 - i0
        if seg_len <= 0:
            continue
        ts = t[i0:i1]
        # Smooth crossfade envelope per chord block
        fade = min(seg_len // 5, int(1.2 * SAMPLE_RATE))
        env = np.ones(seg_len, dtype=np.float64)
        if fade > 0:
            env[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            env[-fade:] = np.cos(np.linspace(0, np.pi / 2, fade)) ** 2

        chord_l = np.zeros(seg_len, dtype=np.float64)
        chord_r = np.zeros(seg_len, dtype=np.float64)
        for k, f in enumerate(freqs):
            detune = 1.0 + (0.0035 if k % 2 == 0 else -0.0035)
            vib = 1.0 + 0.0018 * np.sin(2.0 * np.pi * 0.25 * ts + k)
            w1 = np.sin(2.0 * np.pi * f * vib * ts)
            w2 = 0.35 * np.sin(2.0 * np.pi * f * 2.0 * detune * ts + 0.7)
            pan = 0.35 + 0.3 * (k / max(1, len(freqs) - 1))
            chord_l += (1.0 - pan) * (w1 + w2)
            chord_r += pan * (w1 + w2)

        pad_l[i0:i1] += env * chord_l * 0.10
        pad_r[i0:i1] += env * chord_r * 0.10

    # Deep dramatic sub-pulse (every ~2.4s)
    pulse = np.sin(2.0 * np.pi * 55.0 * t) * (0.45 + 0.55 * np.sin(2.0 * np.pi * 0.42 * t) ** 2)
    pad_l += 0.08 * pulse
    pad_r += 0.08 * pulse

    # Subtle eerie nocturnal air / high string shimmer
    noise_l = _bandpass(rng.standard_normal(n), 240.0, 950.0)
    noise_r = _bandpass(rng.standard_normal(n), 260.0, 980.0)
    pad_l += 0.028 * noise_l
    pad_r += 0.028 * noise_r

    # Gentle high-cut & master fade in/out
    pad_l = _lowpass(pad_l, 2400.0)
    pad_r = _lowpass(pad_r, 2400.0)
    master_fade_in = min(n // 4, int(1.5 * SAMPLE_RATE))
    master_fade_out = min(n // 4, int(2.0 * SAMPLE_RATE))
    pad_l[:master_fade_in] *= np.linspace(0.0, 1.0, master_fade_in)
    pad_r[:master_fade_in] *= np.linspace(0.0, 1.0, master_fade_in)
    pad_l[-master_fade_out:] *= np.linspace(1.0, 0.0, master_fade_out)
    pad_r[-master_fade_out:] *= np.linspace(1.0, 0.0, master_fade_out)

    _write_stereo_wav(out_path, pad_l, pad_r)
    return out_path


def build_sfx_track(
    out_path: Path,
    cues: list[tuple[float, str, float]],
    total_duration_sec: float,
) -> Path:
    """Render synchronized SFX events along the global episode timeline.

    ``cues`` is a list of ``(start_sec, cue_name, shot_duration_sec)``.
    """
    n = int(round(total_duration_sec * SAMPLE_RATE))
    sfx_l = np.zeros(n, dtype=np.float64)
    sfx_r = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(108)

    for start_sec, cue, dur_sec in cues:
        if not cue or cue == "none":
            continue
        i0 = max(0, int(round(start_sec * SAMPLE_RATE)))

        if cue == "wind_ominous":
            length = min(n - i0, int(round(min(dur_sec, 4.5) * SAMPLE_RATE)))
            if length > 100:
                tl = np.linspace(0.0, length / SAMPLE_RATE, length)
                env = np.sin(np.pi * tl / (length / SAMPLE_RATE)) ** 1.4
                wind_l = _bandpass(rng.standard_normal(length), 180.0, 750.0) * env * 0.16
                wind_r = _bandpass(rng.standard_normal(length), 200.0, 780.0) * env * 0.16
                sfx_l[i0 : i0 + length] += wind_l
                sfx_r[i0 : i0 + length] += wind_r

        elif cue == "meteor_impact":
            # Whoosh rising + sub-bass boom around +1.4s into the shot
            offset_boom = i0 + int(1.1 * SAMPLE_RATE)
            length = min(n - offset_boom, int(3.2 * SAMPLE_RATE))
            if length > 100:
                tl = np.linspace(0.0, length / SAMPLE_RATE, length)
                sweep_freq = 160.0 * np.exp(-3.2 * tl) + 38.0
                boom = np.sin(2.0 * np.pi * np.cumsum(sweep_freq) / SAMPLE_RATE) * np.exp(-1.5 * tl)
                crackle = _lowpass(rng.standard_normal(length), 1400.0) * np.exp(-2.6 * tl)
                sig = 0.28 * boom + 0.14 * crackle
                sfx_l[offset_boom : offset_boom + length] += sig
                sfx_r[offset_boom : offset_boom + length] += sig

        elif cue == "compass_chime":
            # Shimmering metallic harmonic chime when the antique brass compass glows
            offset_chime = i0 + int(0.45 * SAMPLE_RATE)
            length = min(n - offset_chime, int(2.6 * SAMPLE_RATE))
            if length > 100:
                tl = np.linspace(0.0, length / SAMPLE_RATE, length)
                freqs = [880.0, 1318.5, 1760.0, 2637.0]
                chime = sum(
                    (0.55 ** k) * np.sin(2.0 * np.pi * f * tl) * np.exp(-(1.4 + 0.4 * k) * tl)
                    for k, f in enumerate(freqs)
                )
                sfx_l[offset_chime : offset_chime + length] += 0.08 * chime
                sfx_r[offset_chime : offset_chime + length] += 0.08 * chime

        elif cue == "portal_hum":
            # Deep resonant sub-harmonic awakening rumble for the final cliffhanger gate
            length = min(n - i0, int(round(dur_sec * SAMPLE_RATE)))
            if length > 100:
                tl = np.linspace(0.0, length / SAMPLE_RATE, length)
                env = np.minimum(1.0, tl / 1.0) * np.exp(-0.22 * np.maximum(0.0, tl - 1.5))
                hum = (
                    0.55 * np.sin(2.0 * np.pi * 65.4 * tl)
                    + 0.30 * np.sin(2.0 * np.pi * 98.0 * tl)
                    + 0.15 * np.sin(2.0 * np.pi * 130.8 * tl)
                ) * env
                sfx_l[i0 : i0 + length] += 0.16 * hum
                sfx_r[i0 : i0 + length] += 0.16 * hum

    _write_stereo_wav(out_path, sfx_l, sfx_r)
    return out_path
