"""Driving 136 BPM action-scene BGM + combat whoosh/slash/impact SFX generator."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLE_RATE = 48_000


def _write_wav(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.clip(np.stack([left, right], axis=1), -0.98, 0.98)
    pcm = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _lowpass(sig: np.ndarray, cutoff: float) -> np.ndarray:
    b, a = butter(3, cutoff / (0.5 * SAMPLE_RATE), btype="low")
    return filtfilt(b, a, sig)


def _bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b, a = butter(2, [lo / (0.5 * SAMPLE_RATE), hi / (0.5 * SAMPLE_RATE)], btype="band")
    return filtfilt(b, a, sig)


def generate_action_bgm(out_path: Path, total_sec: float, battle_start_sec: float, outro_start_sec: float) -> Path:
    """Synthesize tense stealth intro -> 136 BPM battle synth percussion -> calm aftermath."""
    n = int(round(total_sec * SAMPLE_RATE))
    t = np.linspace(0.0, total_sec, n, endpoint=False)
    rng = np.random.default_rng(777)

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # 1. Atmospheric dark drone bed across full timeline
    drone = 0.08 * np.sin(2 * np.pi * 73.42 * t) + 0.05 * np.sin(2 * np.pi * 110.0 * t)
    left += drone
    right += drone

    # 2. Driving 136 BPM battle percussion + staccato synth bass ostinato
    beat_sec = 60.0 / 136.0
    sub_step = beat_sec / 2.0  # 8th-note driving pulse
    bass_notes = [73.42, 73.42, 87.31, 73.42, 98.00, 73.42, 110.00, 98.00]
    arp_notes = [293.66, 349.23, 440.00, 523.25, 440.00, 349.23, 329.63, 293.66]

    cur = max(0.0, battle_start_sec)
    step_idx = 0
    while cur < outro_start_sec:
        i0 = int(round(cur * SAMPLE_RATE))
        # Kick on every quarter beat (even 8th steps)
        if step_idx % 2 == 0:
            k_len = min(n - i0, int(0.22 * SAMPLE_RATE))
            if k_len > 20:
                tk = np.linspace(0.0, k_len / SAMPLE_RATE, k_len)
                freq = 125.0 * np.exp(-18.0 * tk) + 44.0
                kick = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-10.5 * tk)
                left[i0 : i0 + k_len] += 0.24 * kick
                right[i0 : i0 + k_len] += 0.24 * kick

        # Snare/hat snap on off-beats
        if step_idx % 4 == 2:
            s_len = min(n - i0, int(0.14 * SAMPLE_RATE))
            if s_len > 20:
                ts = np.linspace(0.0, s_len / SAMPLE_RATE, s_len)
                snare = _bandpass(rng.standard_normal(s_len), 900.0, 5500.0) * np.exp(-22.0 * ts)
                left[i0 : i0 + s_len] += 0.11 * snare
                right[i0 : i0 + s_len] += 0.11 * snare

        # Staccato driving bass & tense minor arp
        n_len = min(n - i0, int(sub_step * 0.92 * SAMPLE_RATE))
        if n_len > 20:
            tn = np.linspace(0.0, n_len / SAMPLE_RATE, n_len)
            f_bass = bass_notes[step_idx % len(bass_notes)]
            f_arp = arp_notes[step_idx % len(arp_notes)]
            bass = (np.sin(2 * np.pi * f_bass * tn) + 0.4 * np.sin(4 * np.pi * f_bass * tn)) * np.exp(-6.5 * tn)
            arp = np.sin(2 * np.pi * f_arp * tn) * np.exp(-9.0 * tn)
            pan = 0.38 if step_idx % 2 == 0 else 0.62
            left[i0 : i0 + n_len] += 0.13 * bass + 0.08 * (1 - pan) * arp
            right[i0 : i0 + n_len] += 0.13 * bass + 0.08 * pan * arp

        cur += sub_step
        step_idx += 1

    left = _lowpass(left, 6500.0)
    right = _lowpass(right, 6500.0)
    fade_out = min(n // 4, int(1.6 * SAMPLE_RATE))
    left[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    right[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    _write_wav(out_path, left, right)
    return out_path


def generate_action_sfx(
    out_path: Path,
    timeline: list[tuple[str, float, float]],
    total_sec: float,
) -> Path:
    """Create crisp combat dash whooshes, katana energy slashes, impact booms, and chimes."""
    n = int(round(total_sec * SAMPLE_RATE))
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(909)

    for shot_id, start_s, dur_s in timeline:
        i0 = max(0, int(round(start_s * SAMPLE_RATE)))

        if shot_id == "action_002":
            # Sharp tension riser sting
            length = min(n - i0, int(1.2 * SAMPLE_RATE))
            if length > 50:
                tl = np.linspace(0.0, length / SAMPLE_RATE, length)
                freq = 140.0 + 480.0 * (tl / (length / SAMPLE_RATE)) ** 2
                riser = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * (tl / (length / SAMPLE_RATE)) * 0.14
                left[i0 : i0 + length] += riser
                right[i0 : i0 + length] += riser

        elif shot_id == "action_003":
            # Dash whoosh at +0.25s + sharp slash & heavy sub-boom at +1.15s
            w_idx = min(n - 1, i0 + int(0.25 * SAMPLE_RATE))
            w_len = min(n - w_idx, int(0.55 * SAMPLE_RATE))
            if w_len > 50:
                tw = np.linspace(0.0, w_len / SAMPLE_RATE, w_len)
                whoosh = _bandpass(rng.standard_normal(w_len), 400.0, 3200.0) * np.sin(math.pi * tw / (w_len / SAMPLE_RATE)) ** 1.5
                left[w_idx : w_idx + w_len] += 0.24 * whoosh
                right[w_idx : w_idx + w_len] += 0.24 * whoosh

            s_idx = min(n - 1, i0 + int(1.12 * SAMPLE_RATE))
            s_len = min(n - s_idx, int(1.5 * SAMPLE_RATE))
            if s_len > 50:
                ts = np.linspace(0.0, s_len / SAMPLE_RATE, s_len)
                # Metallic slash ring + sub-bass impact shockwave
                slash_hi = (
                    np.sin(2 * np.pi * 2420.0 * ts) * np.exp(-8.5 * ts)
                    + 0.6 * np.sin(2 * np.pi * 3780.0 * ts) * np.exp(-11.0 * ts)
                    + _bandpass(rng.standard_normal(s_len), 1800.0, 6800.0) * np.exp(-15.0 * ts)
                )
                boom_freq = 175.0 * np.exp(-9.0 * ts) + 38.0
                boom = np.sin(2 * np.pi * np.cumsum(boom_freq) / SAMPLE_RATE) * np.exp(-3.2 * ts)
                hit = 0.20 * slash_hi + 0.36 * boom
                left[s_idx : s_idx + s_len] += hit
                right[s_idx : s_idx + s_len] += hit

        elif shot_id == "action_004":
            # Power surge + resonant crystal/energy burst at +1.0s
            b_idx = min(n - 1, i0 + int(0.95 * SAMPLE_RATE))
            b_len = min(n - b_idx, int(1.8 * SAMPLE_RATE))
            if b_len > 50:
                tb = np.linspace(0.0, b_len / SAMPLE_RATE, b_len)
                burst = (
                    0.30 * np.sin(2 * np.pi * np.cumsum(150.0 * np.exp(-5.0 * tb) + 46.0) / SAMPLE_RATE) * np.exp(-2.4 * tb)
                    + 0.10 * np.sin(2 * np.pi * 1318.5 * tb) * np.exp(-3.5 * tb)
                    + 0.14 * _lowpass(rng.standard_normal(b_len), 2200.0) * np.exp(-4.5 * tb)
                )
                left[b_idx : b_idx + b_len] += burst
                right[b_idx : b_idx + b_len] += burst

    _write_wav(out_path, left, right)
    return out_path
