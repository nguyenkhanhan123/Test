"""Step 3: Audio-First TTS generation, WordBoundary timestamps, and duration locking."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import edge_tts

from media_utils import probe_mp3_duration, quantize_to_fps, run_ffmpeg
from schema import EpisodeScript, Shot, VoiceSpec, WordToken

TICKS_PER_SECOND = 10_000_000.0


def _resolve_voice(shot: Shot, episode: EpisodeScript) -> VoiceSpec:
    if shot.voice_spec is not None:
        return shot.voice_spec
    if shot.character in episode.characters:
        return episode.characters[shot.character]
    voice_id = shot.voice_ref.split(":", 1)[1] if ":" in shot.voice_ref else "vi-VN-NamMinhNeural"
    return VoiceSpec(engine="edge_tts", voice_id=voice_id)


async def _synthesize_edge_tts(
    text: str,
    spec: VoiceSpec,
    mp3_path: Path,
    retries: int = 3,
) -> list[WordToken]:
    """Synthesize speech with Edge TTS and capture WordBoundary events."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            comm = edge_tts.Communicate(
                text=text,
                voice=spec.voice_id,
                rate=spec.rate,
                volume=spec.volume,
                pitch=spec.pitch,
                boundary="WordBoundary",
            )
            audio_chunks = bytearray()
            tokens: list[WordToken] = []
            async for item in comm.stream():
                kind = item.get("type")
                if kind == "audio":
                    audio_chunks.extend(item["data"])
                elif kind == "WordBoundary":
                    start = float(item["offset"]) / TICKS_PER_SECOND
                    dur = float(item["duration"]) / TICKS_PER_SECOND
                    word = str(item.get("text", "")).strip()
                    if word:
                        tokens.append(WordToken(word=word, start=start, end=start + dur))
            if not audio_chunks:
                raise RuntimeError("Edge TTS returned empty audio stream")
            mp3_path.write_bytes(bytes(audio_chunks))
            return tokens
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(1.5 * attempt)
    raise RuntimeError(f"Edge TTS failed after {retries} attempts: {last_err}")


def _build_padded_wav(
    raw_mp3: Path,
    out_wav: Path,
    lead_in_sec: float,
    total_duration_sec: float,
) -> None:
    """Convert speech MP3 to 48kHz stereo WAV padded to exact frame-locked length."""
    delay_ms = int(round(lead_in_sec * 1000.0))
    run_ffmpeg(
        [
            "-i",
            str(raw_mp3),
            "-af",
            (
                f"aresample=48000,"
                f"adelay={delay_ms}|{delay_ms},"
                f"apad=whole_dur={total_duration_sec:.6f},"
                f"atrim=0:{total_duration_sec:.6f}"
            ),
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )


async def process_episode_audio(base_dir: Path, script_path: Path) -> EpisodeScript:
    """Generate TTS audio and word timestamps for every shot and update script.json."""
    episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    audio_dir = base_dir / "assets" / "audio"
    sub_dir = base_dir / "assets" / "subtitles"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    for shot in episode.shots:
        spec = _resolve_voice(shot, episode)
        raw_mp3 = audio_dir / f"{shot.shot_id}_raw.mp3"
        padded_wav = audio_dir / f"{shot.shot_id}.wav"
        words_json = sub_dir / f"{shot.shot_id}_words.json"

        tokens = await _synthesize_edge_tts(shot.dialogue, spec, raw_mp3)
        speech_dur = probe_mp3_duration(raw_mp3)

        # Fallback safeguard if a voice omits boundaries unexpectedly
        if not tokens:
            words = [w for w in shot.dialogue.split() if w]
            step = speech_dur / max(1, len(words))
            tokens = [
                WordToken(word=w, start=i * step, end=(i + 1) * step)
                for i, w in enumerate(words)
            ]

        target_total = shot.lead_in_sec + speech_dur + shot.tail_pad_sec
        _, quantized_dur = quantize_to_fps(target_total, episode.fps)

        _build_padded_wav(raw_mp3, padded_wav, shot.lead_in_sec, quantized_dur)

        # Shift local word tokens by lead_in_sec so they match the padded shot clip
        shifted_tokens = [
            WordToken(
                word=t.word,
                start=round(t.start + shot.lead_in_sec, 4),
                end=round(min(t.end + shot.lead_in_sec, quantized_dur - 0.04), 4),
            )
            for t in tokens
        ]
        words_json.write_text(
            json.dumps([t.model_dump() for t in shifted_tokens], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        shot.audio_path = f"assets/audio/{shot.shot_id}.wav"
        shot.speech_duration_sec = round(speech_dur, 4)
        shot.duration_sec = round(quantized_dur, 6)
        print(
            f"[{shot.shot_id}] voice={spec.voice_id} speech={speech_dur:.2f}s "
            f"-> locked_clip={quantized_dur:.3f}s ({len(shifted_tokens)} words)"
        )

    script_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    total_dur = sum(s.duration_sec or 0.0 for s in episode.shots)
    print(f"Updated {script_path} | Total episode duration: {total_dur:.2f}s")
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Audio-First TTS and word timestamps")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "script.json",
        help="Path to script.json",
    )
    args = parser.parse_args()
    base_dir = args.script.resolve().parent
    asyncio.run(process_episode_audio(base_dir, args.script.resolve()))


if __name__ == "__main__":
    main()
