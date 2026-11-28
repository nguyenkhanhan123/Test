"""Action Scene Prototype Runner (Emotional Voice + Combat Choreography VFX + Colab Override).

Uses the extracted Minh Khang & Linh Chi sprites + keyframes from earlier to render
an expressive 5-shot action prototype (output/action_pilot_demo.mp4), while
automatically picking up real Wan 2.2 clips from assets/video_ai/ when rendered on Colab.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
from pathlib import Path

import edge_tts
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from action_sfx_bgm import generate_action_bgm, generate_action_sfx
from media_utils import count_video_frames, get_ffmpeg_exe, probe_media_duration, probe_mp3_duration, quantize_to_fps, run_ffmpeg
from schema import EmotionTag, EpisodeScript, Shot, ShotType, VoiceSpec, WordToken
from step6_subtitles import build_subtitles

TICKS_PER_SECOND = 10_000_000.0


def build_action_demo_script() -> EpisodeScript:
    """Create a tight 5-shot emotional action encounter using Minh Khang & Linh Chi."""
    voices = {
        "minh_khang_whisper": VoiceSpec(
            engine="edge_tts", voice_id="vi-VN-NamMinhNeural", rate="-10%", pitch="-5Hz", volume="+0%"
        ),
        "linh_chi_tense": VoiceSpec(
            engine="edge_tts", voice_id="vi-VN-HoaiMyNeural", rate="+11%", pitch="+4Hz", volume="+0%"
        ),
        "minh_khang_action": VoiceSpec(
            engine="edge_tts", voice_id="vi-VN-NamMinhNeural", rate="+14%", pitch="+7Hz", volume="+0%"
        ),
        "linh_chi_action": VoiceSpec(
            engine="edge_tts", voice_id="vi-VN-HoaiMyNeural", rate="+12%", pitch="+5Hz", volume="+0%"
        ),
        "minh_khang_calm": VoiceSpec(
            engine="edge_tts", voice_id="vi-VN-NamMinhNeural", rate="-6%", pitch="-2Hz", volume="+0%"
        ),
    }

    shots = [
        Shot(
            shot_id="shot_101",
            character="minh_khang",
            character_lora="nv_minhkhang_lora_v1.safetensors",
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=voices["minh_khang_whisper"],
            shot_type=ShotType.KEN_BURNS_ZOOM_IN,
            emotion=EmotionTag.WHISPER,
            dialogue="Suỵt... đừng phát ra tiếng động. Cánh cổng đang mở ra rồi.",
            image_prompt="nv_minhkhang, rear view facing awakened glowing crimson portal in misty forest, tense stealth atmosphere",
            lead_in_sec=0.25,
            tail_pad_sec=0.35,
            keyframe_path="assets/keyframes/shot_006.png",
        ),
        Shot(
            shot_id="shot_102",
            character="linh_chi",
            character_lora="linh_chi_v1.safetensors",
            voice_ref="edge_tts:vi-VN-HoaiMyNeural",
            voice_spec=voices["linh_chi_tense"],
            shot_type=ShotType.VIDEO_AI,
            emotion=EmotionTag.TENSE,
            dialogue="Khang! Phía trước có dị năng đang lao thẳng tới!",
            image_prompt="Linh Chi alerting Minh Khang in front of rising crimson forest beacon, urgent action lighting",
            motion_prompt="rapid whip zoom, urgent camera tremor, glowing red sparks surging",
            lead_in_sec=0.15,
            tail_pad_sec=0.25,
            keyframe_path="assets/keyframes/shot_005.png",
        ),
        Shot(
            shot_id="shot_103",
            character="minh_khang",
            character_lora="nv_minhkhang_lora_v1.safetensors",
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=voices["minh_khang_action"],
            shot_type=ShotType.VIDEO_AI,
            emotion=EmotionTag.ANGRY_ACTION,
            dialogue="Lùi lại phía sau! Để tôi chặn đòn này!",
            image_prompt="nv_minhkhang, high-speed forward dash slash with crimson energy arc and shockwave, anime action scene",
            motion_prompt="fast dynamic forward dash, blazing crimson energy slash, impact shockwave, camera shake",
            lead_in_sec=0.12,
            tail_pad_sec=0.55,
            keyframe_path="assets/keyframes/shot_003.png",
        ),
        Shot(
            shot_id="shot_104",
            character="linh_chi",
            character_lora="linh_chi_v1.safetensors",
            voice_ref="edge_tts:vi-VN-HoaiMyNeural",
            voice_spec=voices["linh_chi_action"],
            shot_type=ShotType.VIDEO_AI,
            emotion=EmotionTag.ANGRY_ACTION,
            dialogue="Ngay lúc này! Phá vỡ kết giới đi!",
            image_prompt="Minh Khang and Linh Chi launching dual counter burst against crimson portal, golden runes and red lightning",
            motion_prompt="intense dual counterattack burst, spinning golden compass aura, speed lines, dramatic camera push",
            lead_in_sec=0.12,
            tail_pad_sec=0.55,
            keyframe_path="assets/keyframes/shot_006.png",
        ),
        Shot(
            shot_id="shot_105",
            character="minh_khang",
            character_lora="nv_minhkhang_lora_v1.safetensors",
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=voices["minh_khang_calm"],
            shot_type=ShotType.KEN_BURNS_ZOOM_OUT,
            emotion=EmotionTag.CALM,
            dialogue="Tạm thời đẩy lùi được rồi... nhưng trận chiến thật sự mới chỉ bắt đầu.",
            image_prompt="nv_minhkhang catching breath on stone lookout under fading red storm sky, calm aftermath",
            lead_in_sec=0.30,
            tail_pad_sec=1.00,
            keyframe_path="assets/keyframes/shot_001.png",
        ),
    ]

    return EpisodeScript(
        episode_id="ep_action_demo",
        title="Thử Nghiệm Cảnh Hành Động & Giọng Biểu Cảm",
        fps=24,
        width=1280,
        height=720,
        style_lora="anime_cinematic_v1.safetensors",
        characters={
            "minh_khang": voices["minh_khang_action"],
            "linh_chi": voices["linh_chi_action"],
        },
        shots=shots,
    )


async def _synth_emotional_tts(base_dir: Path, script_path: Path) -> EpisodeScript:
    episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    audio_dir = base_dir / "assets" / "audio"
    sub_dir = base_dir / "assets" / "subtitles"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    for shot in episode.shots:
        spec = shot.voice_spec
        assert spec is not None
        raw_mp3 = audio_dir / f"{shot.shot_id}_raw.mp3"
        padded_wav = audio_dir / f"{shot.shot_id}.wav"
        words_json = sub_dir / f"{shot.shot_id}_words.json"

        chunks = bytearray()
        tokens: list[WordToken] = []
        last_err: Exception | None = None
        for attempt in range(1, 5):
            try:
                comm = edge_tts.Communicate(
                    text=shot.dialogue,
                    voice=spec.voice_id,
                    rate=spec.rate,
                    pitch=spec.pitch,
                    volume=spec.volume,
                    boundary="WordBoundary",
                )
                chunks.clear()
                tokens.clear()
                async for ev in comm.stream():
                    if ev.get("type") == "audio":
                        chunks.extend(ev["data"])
                    elif ev.get("type") == "WordBoundary":
                        s = float(ev["offset"]) / TICKS_PER_SECOND
                        d = float(ev["duration"]) / TICKS_PER_SECOND
                        w = str(ev.get("text", "")).strip()
                        if w:
                            tokens.append(WordToken(word=w, start=s, end=s + d))
                if chunks:
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(1.5 * attempt)
        if not chunks:
            raise RuntimeError(f"Edge TTS failed for {shot.shot_id}: {last_err}")
        raw_mp3.write_bytes(bytes(chunks))
        speech_dur = probe_mp3_duration(raw_mp3)
        _, q_dur = quantize_to_fps(shot.lead_in_sec + speech_dur + shot.tail_pad_sec, episode.fps)

        # Emotion-matched EQ/dynamics so whisper vs combat shout feel distinct
        delay_ms = int(round(shot.lead_in_sec * 1000))
        if shot.emotion == EmotionTag.WHISPER:
            fx = "volume=0.78,lowpass=f=4600,acompressor=threshold=-22dB:ratio=2.5"
        elif shot.emotion == EmotionTag.ANGRY_ACTION:
            fx = "highpass=f=95,acompressor=threshold=-16dB:ratio=4.0:makeup=3dB,volume=1.15"
        else:
            fx = "highpass=f=80"

        run_ffmpeg(
            [
                "-i", str(raw_mp3),
                "-af", f"aresample=48000,{fx},adelay={delay_ms}|{delay_ms},apad=whole_dur={q_dur:.6f},atrim=0:{q_dur:.6f}",
                "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le",
                str(padded_wav),
            ]
        )
        shifted = [
            WordToken(
                word=t.word,
                start=round(t.start + shot.lead_in_sec, 4),
                end=round(min(t.end + shot.lead_in_sec, q_dur - 0.04), 4),
            )
            for t in tokens
        ]
        words_json.write_text(json.dumps([x.model_dump() for x in shifted], ensure_ascii=False, indent=2), encoding="utf-8")
        shot.audio_path = f"assets/audio/{shot.shot_id}.wav"
        shot.speech_duration_sec = round(speech_dur, 4)
        shot.duration_sec = round(q_dur, 6)
        print(f"[{shot.shot_id}] emotion={shot.emotion.value} speech={speech_dur:.2f}s -> {q_dur:.3f}s")

    script_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    return episode


def _draw_speed_lines(canvas: Image.Image, center: tuple[int, int], seed: int, strength: float) -> Image.Image:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng(seed)
    w, h = canvas.size
    cx, cy = center
    for _ in range(44):
        ang = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(220, 420)
        r1 = 900
        width_px = int(rng.integers(2, 6))
        alpha = int(min(220, rng.integers(75, 165) * strength))
        x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        draw.line([(x0, y0), (x1, y1)], fill=(255, 210, 200, alpha), width=width_px)
    return Image.alpha_composite(canvas.convert("RGBA"), overlay)


def _render_combat_vfx_clip(
    base_dir: Path,
    shot: Shot,
    width: int,
    height: int,
    fps: int,
    out_clip: Path,
) -> int:
    """Render dynamic anime action choreography using extracted character sprites + VFX."""
    n_frames = max(1, int(round((shot.duration_sec or 3.5) * fps)))
    raw_bg = Image.open(base_dir / shot.keyframe_path).convert("RGB")
    if shot.shot_id == "shot_104":
        # Crop upper portal vortex from shot_006 so foreground sprites never duplicate rear silhouettes
        raw_bg = raw_bg.crop((140, 0, 1236, 430))
    bg_src = raw_bg.resize((int(width * 1.35), int(height * 1.35)), Image.Resampling.LANCZOS)
    mk_sprite = Image.open(base_dir / "assets" / "sprites" / "minh_khang.png").convert("RGBA")
    lc_sprite = Image.open(base_dir / "assets" / "sprites" / "linh_chi.png").convert("RGBA")

    exe = get_ffmpeg_exe()
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-i", str(base_dir / shot.audio_path),
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-shortest", str(out_clip),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    rng = np.random.default_rng(abs(hash(shot.shot_id)) % (2**31))
    bw, bh = bg_src.size

    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        t = i / float(fps)

        # Rapid dynamic camera pan/zoom + decay shake around impact (~1.15s)
        impact_dt = t - 1.15
        shake = math.exp(-5.5 * impact_dt) * 18.0 if impact_dt >= 0 else 0.0
        sx = math.sin(t * 58.0) * shake
        sy = math.cos(t * 47.0) * shake

        if shot.shot_id == "shot_102":
            zoom = 1.05 + 0.22 * (u ** 0.7)
            cx = bw * (0.52 - 0.05 * u) + math.sin(t * 18) * 4
            cy = bh * 0.48 + math.cos(t * 14) * 3
        elif shot.shot_id == "shot_103":
            zoom = 1.08 + 0.16 * u
            cx = bw * (0.38 + 0.24 * u) + sx
            cy = bh * (0.52 - 0.06 * u) + sy
        else:  # shot_104 duo surge
            zoom = 1.04 + 0.25 * (u ** 0.85)
            cx = bw * 0.50 + sx
            cy = bh * (0.54 - 0.08 * u) + sy

        cw, ch = bw / zoom, bh / zoom
        x0 = max(0.0, min(bw - cw, cx - cw / 2.0))
        y0 = max(0.0, min(bh - ch, cy - ch / 2.0))
        frame_rgba = bg_src.transform(
            (width, height), Image.Transform.EXTENT, (x0, y0, x0 + cw, y0 + ch), resample=Image.Resampling.BICUBIC
        ).convert("RGBA")

        if shot.shot_id == "shot_103":
            # Minh Khang high-speed dash from left -> center-right with afterimage ghost trails
            dash_u = min(1.0, u * 1.45)
            ease = 1.0 - (1.0 - dash_u) ** 3
            scale_h = int(height * (0.98 + 0.18 * ease))
            mk_resized = mk_sprite.resize(
                (int(mk_sprite.width * scale_h / mk_sprite.height), scale_h), Image.Resampling.LANCZOS
            )
            mk_x = int(-140 + ease * 580)
            mk_y = int(height * 0.12 - 20 * math.sin(dash_u * math.pi))

            # Ghost afterimages during fast dash
            if dash_u < 0.92:
                for g_idx, lag in enumerate((65, 130)):
                    ghost = mk_resized.copy()
                    r, g, b, a = ghost.split()
                    a = a.point(lambda p, f=(0.35 - 0.12 * g_idx): int(p * f))
                    ghost = Image.merge("RGBA", (r.point(lambda p: min(255, int(p * 1.3))), g, b, a))
                    frame_rgba.alpha_composite(ghost, (max(-400, mk_x - lag), mk_y + 8 * (g_idx + 1)))

            frame_rgba.alpha_composite(mk_resized, (mk_x, mk_y))

            # Animated tapered crescent anime energy slash arc (0.92s -> 1.72s)
            if 0.92 <= t <= 1.72:
                su = (t - 0.92) / 0.80
                slash_glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                slash_core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                gd = ImageDraw.Draw(slash_glow)
                cd = ImageDraw.Draw(slash_core)
                fade = max(0.0, 1.0 - max(0.0, (su - 0.35) / 0.65))
                n_pts = 32
                max_u = min(1.0, su * 1.95)
                upper_glow, lower_glow = [], []
                upper_core, lower_core = [], []
                for k in range(n_pts):
                    ku = (k / (n_pts - 1)) * max_u
                    px = 210.0 + ku * 880.0
                    py = 560.0 - ku * 410.0 - math.sin(ku * math.pi) * 145.0
                    taper = math.sin(min(1.0, ku / max(0.05, max_u)) * math.pi) ** 1.15
                    wg = 24.0 * taper * fade
                    wc = 8.0 * taper * fade
                    upper_glow.append((px - wg * 0.45, py - wg))
                    lower_glow.append((px + wg * 0.45, py + wg))
                    upper_core.append((px - wc * 0.45, py - wc))
                    lower_core.append((px + wc * 0.45, py + wc))
                if len(upper_glow) >= 3:
                    gd.polygon(upper_glow + list(reversed(lower_glow)), fill=(255, 45, 70, int(220 * fade)))
                    cd.polygon(upper_core + list(reversed(lower_core)), fill=(255, 248, 240, int(250 * fade)))
                if su >= 0.22:
                    ru = (su - 0.22) / 0.78
                    rad = int(35 + ru * 420)
                    ra = int(220 * (1.0 - ru))
                    gd.ellipse([650 - rad, 290 - rad, 650 + rad, 290 + rad], outline=(255, 110, 90, ra), width=10)
                frame_rgba = Image.alpha_composite(frame_rgba, slash_glow.filter(ImageFilter.GaussianBlur(5)))
                frame_rgba = Image.alpha_composite(frame_rgba, slash_core)

            frame_rgba = _draw_speed_lines(frame_rgba, (720, 320), i, 1.0)

        elif shot.shot_id == "shot_104":
            # Duo counter-strike: Minh Khang surging left + Linh Chi channeling compass right
            mk_h = int(height * (0.92 + 0.16 * u))
            lc_h = int(height * (0.86 + 0.14 * u))
            mk_r = mk_sprite.resize((int(mk_sprite.width * mk_h / mk_sprite.height), mk_h), Image.Resampling.LANCZOS)
            lc_r = lc_sprite.resize((int(lc_sprite.width * lc_h / lc_sprite.height), lc_h), Image.Resampling.LANCZOS)

            aura = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ad = ImageDraw.Draw(aura)
            lc_x, lc_y = int(820 - 50 * u - sx * 0.4), int(height * 0.20)
            compass_cx = lc_x + int(lc_r.width * 0.48)
            compass_cy = lc_y + int(lc_r.height * 0.62)
            pulse_r = int(55 + 35 * abs(math.sin(t * 8.0)))
            ad.ellipse([compass_cx - pulse_r, compass_cy - pulse_r, compass_cx + pulse_r, compass_cy + pulse_r], fill=(255, 190, 75, 95))
            ad.ellipse([compass_cx - pulse_r - 22, compass_cy - pulse_r - 22, compass_cx + pulse_r + 22, compass_cy + pulse_r + 22], outline=(255, 120, 70, 140), width=4)
            # Dynamic red energy shockwave beams shooting from center portal
            for k in range(8):
                ang = (t * 2.8 + k * math.pi / 4.0)
                r_out = int(190 + 280 * u)
                ad.line([(640, 290), (640 + r_out * math.cos(ang), 290 + r_out * math.sin(ang))], fill=(255, 65, 90, 165), width=6)
            frame_rgba = Image.alpha_composite(frame_rgba, aura.filter(ImageFilter.GaussianBlur(2.5)))

            frame_rgba.alpha_composite(mk_r, (int(90 + 75 * u + sx * 0.4), int(height * 0.16)))
            frame_rgba.alpha_composite(lc_r, (lc_x, lc_y))
            frame_rgba = _draw_speed_lines(frame_rgba, (640, 300), i + 100, 1.10)

        else:
            # shot_102 rapid alert sparks
            frame_rgba = _draw_speed_lines(frame_rgba, (640, 360), i + 50, 0.55)

        arr = np.asarray(frame_rgba.convert("RGB"), dtype=np.float32)

        # 2-frame hit-stop flash at impact peak (preserves dark contrast)
        if (shot.shot_id == "shot_103" and 1.13 <= t <= 1.21) or (shot.shot_id == "shot_104" and 0.96 <= t <= 1.04):
            arr = arr * 0.72 + np.array([255.0, 185.0, 175.0], dtype=np.float32) * 0.28

        proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"Action VFX render failed: {err}")
    return count_video_frames(out_clip)


def render_action_demo(base_dir: Path, script_path: Path, skip_tts: bool = False) -> Path:
    if not skip_tts:
        asyncio.run(_synth_emotional_tts(base_dir, script_path))
    episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    from step6_render_local import _escape_ffmpeg_filter_path, _normalize_external_video_ai_clip, _prepare_16x9_image, _render_ken_burns_clip

    prep_dir = base_dir / "assets" / "keyframes_16x9"
    clips_dir = base_dir / "assets" / "clips"
    out_dir = base_dir / "output"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered_clips: list[Path] = []
    timeline: list[tuple[str, float, float]] = []
    cur_t = 0.0

    for idx, shot in enumerate(episode.shots):
        norm_img = prep_dir / f"{shot.shot_id}.png"
        _prepare_16x9_image(base_dir / shot.keyframe_path, norm_img, episode.width, episode.height)
        audio_path = base_dir / shot.audio_path
        clip_path = clips_dir / f"{shot.shot_id}.mp4"
        ai_override = base_dir / "assets" / "video_ai" / f"{shot.shot_id}.mp4"

        if shot.shot_type.is_video_ai and ai_override.exists():
            n_frames = _normalize_external_video_ai_clip(
                ai_override, audio_path, clip_path, shot, episode.width, episode.height, episode.fps
            )
            shot.render_mode_used = "wan22_colab_real"
        elif shot.shot_type.is_video_ai:
            n_frames = _render_combat_vfx_clip(base_dir, shot, episode.width, episode.height, episode.fps, clip_path)
            shot.render_mode_used = "local_action_sprite_vfx"
        else:
            n_frames = _render_ken_burns_clip(norm_img, audio_path, clip_path, shot, episode.width, episode.height, episode.fps)
            shot.render_mode_used = shot.shot_type.value

        dur = n_frames / float(episode.fps)
        shot.video_clip_path = f"assets/clips/{shot.shot_id}.mp4"
        shot.duration_sec = round(dur, 6)
        rendered_clips.append(clip_path)
        timeline.append((f"action_{idx + 1:03d}", cur_t, dur))
        cur_t += dur
        print(f"[{idx+1}/{len(episode.shots)}] {shot.shot_id} ({shot.render_mode_used}) -> {n_frames}f ({dur:.3f}s)")

    script_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")

    raw_mp4 = out_dir / "action_raw.mp4"
    inputs: list[str] = []
    streams: list[str] = []
    for i, c in enumerate(rendered_clips):
        inputs.extend(["-i", str(c)])
        streams.append(f"[{i}:v][{i}:a]")
    fade_st = max(0.0, cur_t - 0.7)
    run_ffmpeg([
        *inputs,
        "-filter_complex", f"{''.join(streams)}concat=n={len(rendered_clips)}:v=1:a=1[vc][ac];[vc]fade=t=in:st=0:d=0.30,fade=t=out:st={fade_st:.3f}:d=0.70[vo]",
        "-map", "[vo]", "-map", "[ac]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
        str(raw_mp4),
    ])

    total_dur = probe_media_duration(raw_mp4)
    bgm_wav = base_dir / "assets" / "bgm" / "action_bgm.wav"
    sfx_wav = base_dir / "assets" / "sfx" / "action_sfx.wav"
    battle_start = timeline[1][1]  # battle kicks in at shot_102
    outro_start = timeline[-1][1]  # drops calm on shot_105
    generate_action_bgm(bgm_wav, total_dur, battle_start, outro_start)
    generate_action_sfx(sfx_wav, timeline, total_dur)

    mixed_mp4 = out_dir / "action_mixed.mp4"
    run_ffmpeg([
        "-i", str(raw_mp4), "-i", str(bgm_wav), "-i", str(sfx_wav),
        "-filter_complex",
        (
            "[0:a]asplit=2[vsc][vmix];[1:a]volume=0.58[bg];"
            "[bg][vsc]sidechaincompress=threshold=0.015:ratio=6.5:attack=35:release=320:makeup=1[bgd];"
            "[2:a]volume=0.95[sx];[vmix][bgd][sx]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.95[aout]"
        ),
        "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(mixed_mp4),
    ])

    ass_path = out_dir / "action_demo.ass"
    srt_path = out_dir / "action_demo.srt"
    build_subtitles(base_dir, script_path, ass_path, srt_path, episode=episode)

    final_mp4 = out_dir / "action_pilot_demo.mp4"
    run_ffmpeg([
        "-i", str(mixed_mp4), "-vf", f"ass='{_escape_ffmpeg_filter_path(ass_path)}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(final_mp4),
    ])
    print(f"ACTION DEMO VIDEO READY: {final_mp4} ({total_dur:.2f}s)")
    return final_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--tts-only", action="store_true")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    script_path = base_dir / "script_action_demo.json"
    if (not args.skip_tts and not args.tts_only) or not script_path.exists():
        ep = build_action_demo_script()
        script_path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
    if args.tts_only:
        asyncio.run(_synth_emotional_tts(base_dir, script_path))
        return
    render_action_demo(base_dir, script_path, skip_tts=args.skip_tts)


if __name__ == "__main__":
    main()
