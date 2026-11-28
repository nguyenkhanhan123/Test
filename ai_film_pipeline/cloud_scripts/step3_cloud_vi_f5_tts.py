"""Cloud GPU (Colab T4/L4) Step 3: Multi-Emotion Voice Cloning with danhtran2mind/Vi-F5-TTS + WhisperX.

Uses verified CLI flags for danhtran2mind/Vi-F5-TTS (`f5-tts_infer-cli`) and
paired WhisperX Vietnamese forced alignment (`nguyenvulebinh/wav2vec2-large-vi-vlsp2020`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_utils import probe_media_duration, quantize_to_fps, run_ffmpeg
from schema import EmotionTag, EpisodeScript, Shot, VoiceEmotionSample, WordToken

VI_ALIGN_MODEL = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"


def _pick_emotion_sample(shot: Shot, episode: EpisodeScript) -> VoiceEmotionSample:
    spec = shot.voice_spec or episode.characters.get(shot.character)
    if spec is None or not spec.emotion_bank:
        raise RuntimeError(
            f"Shot {shot.shot_id} ({shot.character}) needs VoiceSpec.emotion_bank populated."
        )
    if shot.emotion in spec.emotion_bank:
        return spec.emotion_bank[shot.emotion]
    return spec.emotion_bank.get(EmotionTag.NEUTRAL) or next(iter(spec.emotion_bank.values()))


def _pad_wav_to_fps(raw_wav: Path, out_wav: Path, lead_in_sec: float, total_sec: float) -> None:
    delay_ms = int(round(lead_in_sec * 1000.0))
    run_ffmpeg(
        [
            "-i", str(raw_wav),
            "-af", f"aresample=48000,adelay={delay_ms}|{delay_ms},apad=whole_dur={total_sec:.6f},atrim=0:{total_sec:.6f}",
            "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )


def clone_episode_voices_vi_f5(
    base_dir: Path,
    script_path: Path,
    model_dir: Path,
) -> None:
    """Synthesize expressive per-scene voice cloning and forced-aligned word timings."""
    import torch
    import whisperx  # type: ignore

    episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    audio_dir = base_dir / "assets" / "audio"
    sub_dir = base_dir / "assets" / "subtitles"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model, align_meta = whisperx.load_align_model(
        language_code="vi",
        device=device,
        model_name=VI_ALIGN_MODEL,
    )

    cfg_file = model_dir / "vi-fine-tuned-f5-tts.yaml"
    ckpt_file = model_dir / "model_last.pt"
    vocab_file = model_dir / "vocab.txt"

    for shot in episode.shots:
        sample = _pick_emotion_sample(shot, episode)
        raw_wav = audio_dir / f"{shot.shot_id}_raw.wav"
        padded_wav = audio_dir / f"{shot.shot_id}.wav"
        words_json = sub_dir / f"{shot.shot_id}_words.json"

        ref_audio = base_dir / sample.ref_audio
        cmd = [
            "f5-tts_infer-cli",
            "--model_cfg", str(cfg_file),
            "--ckpt_file", str(ckpt_file),
            "--vocab_file", str(vocab_file),
            "--ref_audio", str(ref_audio),
            "--ref_text", sample.ref_text,
            "--gen_text", shot.dialogue,
            "--speed", str(sample.speed),
            "--output_dir", str(audio_dir),
            "--output_file", raw_wav.name,
        ]
        subprocess.run(cmd, check=True)

        speech_dur = probe_media_duration(raw_wav)
        _, quantized_dur = quantize_to_fps(
            shot.lead_in_sec + speech_dur + shot.tail_pad_sec, episode.fps
        )
        _pad_wav_to_fps(raw_wav, padded_wav, shot.lead_in_sec, quantized_dur)

        audio_arr = whisperx.load_audio(str(raw_wav))
        segments = [{"text": shot.dialogue, "start": 0.0, "end": speech_dur}]
        aligned = whisperx.align(
            segments, align_model, align_meta, audio_arr, device, return_char_alignments=False
        )

        tokens: list[WordToken] = []
        for w in aligned.get("word_segments", []):
            txt = str(w.get("word", "")).strip()
            if not txt or "start" not in w or "end" not in w:
                continue
            tokens.append(
                WordToken(
                    word=txt,
                    start=round(float(w["start"]) + shot.lead_in_sec, 4),
                    end=round(min(float(w["end"]) + shot.lead_in_sec, quantized_dur - 0.04), 4),
                )
            )
        words_json.write_text(
            json.dumps([t.model_dump() for t in tokens], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shot.audio_path = f"assets/audio/{shot.shot_id}.wav"
        shot.speech_duration_sec = round(speech_dur, 4)
        shot.duration_sec = round(quantized_dur, 6)
        print(f"[{shot.shot_id}] emotion={shot.emotion.value} speech={speech_dur:.2f}s -> {quantized_dur:.3f}s")

    script_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
