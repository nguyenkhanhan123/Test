"""Automated verification checks for schema, subtitle sync, and output media."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysubs2
from media_utils import count_video_frames, probe_media_duration, quantize_to_fps
from schema import EmotionTag, EpisodeScript, VoiceEmotionSample, VoiceSpec, WordToken
from step6_subtitles import _chunk_tokens, _restore_punctuation


def test_emotion_voice_and_action_schema() -> None:
    spec = VoiceSpec(
        engine="vi_f5_tts",
        voice_id="user_inputs/voices/calm.wav",
        emotion_bank={
            EmotionTag.ANGRY_ACTION: VoiceEmotionSample(
                ref_audio="user_inputs/voices/angry_action.wav",
                ref_text="Tránh ra ngay lập tức!",
                speed=1.08,
            )
        },
    )
    assert EmotionTag.ANGRY_ACTION in spec.emotion_bank
    nb_path = Path(__file__).resolve().parents[1] / "cloud_scripts" / "COLAB_PRO_ACTION_TEST.ipynb"
    assert nb_path.exists()
    import json

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 5
    import py_compile

    cloud_dir = Path(__file__).resolve().parents[1] / "cloud_scripts"
    py_compile.compile(str(cloud_dir / "train_sdxl_lora_peft.py"), doraise=True)
    py_compile.compile(str(cloud_dir / "step5_batch_wan22_diffusers.py"), doraise=True)


def test_subtitle_punctuation_and_chunking() -> None:
    tokens = [
        WordToken(word="Khang", start=0.37, end=0.55),
        WordToken(word="nhìn", start=0.55, end=0.84),
        WordToken(word="xem", start=0.84, end=1.20),
        WordToken(word="Chiếc", start=2.05, end=2.21),
        WordToken(word="la", start=2.21, end=2.42),
        WordToken(word="bàn", start=2.42, end=2.69),
    ]
    restored = _restore_punctuation(tokens, "Khang nhìn xem! Chiếc la bàn")
    assert restored[2].word == "xem!"
    chunks = _chunk_tokens(restored)
    assert len(chunks) == 2
    assert [t.word for t in chunks[0]] == ["Khang", "nhìn", "xem!"]
    assert [t.word for t in chunks[1]] == ["Chiếc", "la", "bàn"]


def test_quantize_to_fps() -> None:
    frames, sec = quantize_to_fps(4.50, 24)
    assert frames == 108
    assert abs(sec - 4.5) < 1e-9

    frames2, sec2 = quantize_to_fps(5.01, 24)
    assert frames2 == 121
    assert abs(sec2 - 121 / 24.0) < 1e-9


def test_script_and_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "script.json"
    ep = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    ai_n, kb_n, ratio = ep.dynamic_ratio()
    assert len(ep.shots) == 6
    assert ai_n == 2 and kb_n == 4
    assert 0.25 <= ratio <= 0.40

    # Verify TTS durations & WAV files
    expected_total = 0.0
    for shot in ep.shots:
        assert shot.duration_sec is not None and shot.duration_sec > 0
        assert shot.audio_path is not None
        wav_path = root / shot.audio_path
        assert wav_path.exists(), f"Missing audio: {wav_path}"
        expected_total += shot.duration_sec

    # Verify final MP4 & ASS subtitles if already rendered
    final_mp4 = root / "output" / "pilot_30s.mp4"
    ass_path = root / "output" / "episode_01.ass"
    if final_mp4.exists():
        vid_dur = probe_media_duration(final_mp4)
        n_frames = count_video_frames(final_mp4)
        assert abs(vid_dur - expected_total) < 0.15, f"Duration drift: {vid_dur} vs {expected_total}"
        assert abs(n_frames / ep.fps - expected_total) < 0.15

        subs = pysubs2.load(str(ass_path))
        assert len(subs.events) >= len(ep.shots)
        assert subs.events[0].start >= 250  # respects lead_in_sec on shot_001
        assert subs.events[-1].end <= int(round(vid_dur * 1000)) + 50
        # Ensure strictly non-overlapping sequential subtitle cues
        for prev, nxt in zip(subs.events, subs.events[1:]):
            assert prev.start <= nxt.start
            assert prev.end <= nxt.start, f"Overlapping cues: {prev.text} ({prev.end}) vs {nxt.text} ({nxt.start})"
        print(f"ALL RENDER CHECKS PASSED: {final_mp4.name} ({vid_dur:.2f}s, {n_frames} frames, {len(subs.events)} ASS cues)")
    else:
        print("Schema & audio checks passed (final video not yet rendered).")


    # Verify action demo if rendered
    action_mp4 = root / "output" / "action_pilot_demo.mp4"
    action_script = root / "script_action_demo.json"
    action_ass = root / "output" / "action_demo.ass"
    if action_mp4.exists() and action_script.exists():
        ep_act = EpisodeScript.model_validate_json(action_script.read_text(encoding="utf-8"))
        act_expected = sum(s.duration_sec or 0.0 for s in ep_act.shots)
        act_dur = probe_media_duration(action_mp4)
        act_frames = count_video_frames(action_mp4)
        assert abs(act_dur - act_expected) < 0.15, f"Action drift: {act_dur} vs {act_expected}"
        act_subs = pysubs2.load(str(action_ass))
        for prev, nxt in zip(act_subs.events, act_subs.events[1:]):
            assert prev.start <= nxt.start
            assert prev.end <= nxt.start, f"Action overlap: {prev.text} vs {nxt.text}"
        print(f"ACTION DEMO CHECKS PASSED: {action_mp4.name} ({act_dur:.2f}s, {act_frames} frames, {len(act_subs.events)} ASS cues)")


if __name__ == "__main__":
    test_quantize_to_fps()
    test_subtitle_punctuation_and_chunking()
    test_emotion_voice_and_action_schema()
    test_script_and_outputs()
    print("Tests passed.")
