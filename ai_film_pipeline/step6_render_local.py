"""Step 6b: Render Ken Burns + 2.5D motion shots, mix ducked BGM/SFX, and burn ASS subtitles."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from media_utils import count_video_frames, get_ffmpeg_exe, probe_media_duration, probe_mp3_duration, run_ffmpeg
from schema import EpisodeScript, Shot, ShotType
from sfx_bgm import build_sfx_track, generate_cinematic_bgm
from step6_subtitles import build_subtitles


def _prepare_16x9_image(src_path: Path, dst_path: Path, width: int, height: int) -> None:
    """Center-crop and resize any aspect ratio keyframe cleanly to exact target dimensions."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        rgb = img.convert("RGB")
        target_ratio = width / float(height)
        src_ratio = rgb.width / float(rgb.height)
        if abs(src_ratio - target_ratio) > 0.01:
            if src_ratio > target_ratio:
                new_w = int(round(rgb.height * target_ratio))
                left = (rgb.width - new_w) // 2
                rgb = rgb.crop((left, 0, left + new_w, rgb.height))
            else:
                new_h = int(round(rgb.width / target_ratio))
                top = (rgb.height - new_h) // 2
                rgb = rgb.crop((0, top, rgb.width, top + new_h))
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        rgb.save(dst_path, format="PNG")


def _render_ken_burns_clip(
    image_path: Path,
    audio_path: Path,
    out_clip: Path,
    shot: Shot,
    width: int,
    height: int,
    fps: int,
) -> int:
    """Render a jitter-free 3x-supersampled FFmpeg zoompan clip locked to audio frames."""
    n_frames = max(1, int(round((shot.duration_sec or 4.0) * fps)))
    d_denom = max(1, n_frames - 1)

    if shot.shot_type == ShotType.KEN_BURNS_ZOOM_IN:
        zp = f"z='1+0.13*(on/{d_denom})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif shot.shot_type == ShotType.KEN_BURNS_ZOOM_OUT:
        zp = f"z='1.13-0.13*(on/{d_denom})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif shot.shot_type == ShotType.KEN_BURNS_PAN_RIGHT:
        zp = f"z='1.14':x='(iw-iw/zoom)*(on/{d_denom})':y='ih/2-(ih/zoom/2)'"
    elif shot.shot_type == ShotType.KEN_BURNS_PAN_LEFT:
        zp = f"z='1.14':x='(iw-iw/zoom)*(1-on/{d_denom})':y='ih/2-(ih/zoom/2)'"
    else:  # KEN_BURNS_TILT_UP
        zp = f"z='1.15':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{d_denom})'"

    up_w, up_h = width * 3, height * 3
    vf = (
        f"scale={up_w}:{up_h}:flags=lanczos,"
        f"zoompan={zp}:d={n_frames}:s={width}x{height}:fps={fps},"
        f"setsar=1,format=yuv420p"
    )
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-i",
            str(audio_path),
            "-vf",
            vf,
            "-frames:v",
            str(n_frames),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(out_clip),
        ]
    )
    return count_video_frames(out_clip)


def _render_local_25d_motion_clip(
    image_path: Path,
    audio_path: Path,
    out_clip: Path,
    shot: Shot,
    width: int,
    height: int,
    fps: int,
) -> int:
    """Local CPU 2.5D atmospheric motion fallback when Wan 2.2 GPU clip is absent.

    Combines smooth subpixel camera push/tilt, crimson sky light pulse, drifting
    volumetric mist, and glowing embers piped directly into FFmpeg at 24fps.
    """
    n_frames = max(1, int(round((shot.duration_sec or 4.0) * fps)))
    up_w = int(round(width * 1.25))
    up_h = int(round(height * 1.25))

    with Image.open(image_path) as raw:
        base_img = raw.convert("RGB").resize((up_w, up_h), Image.Resampling.LANCZOS)
        # Subtle warm crimson atmospheric glow layer from bright red channels
        arr_base = np.asarray(base_img, dtype=np.float32)

    red_excess = np.clip((arr_base[:, :, 0] - 0.55 * arr_base[:, :, 2]) / 255.0, 0.0, 1.0)
    glow_mask = Image.fromarray((red_excess * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))
    glow_arr = np.asarray(glow_mask, dtype=np.float32)[:, :, None] / 255.0

    # Deterministic rising embers for dynamic energy
    rng = np.random.default_rng(abs(hash(shot.shot_id)) % (2**31))
    n_embers = 42
    ex0 = rng.uniform(0.05 * width, 0.95 * width, n_embers)
    ey0 = rng.uniform(0.10 * height, 1.15 * height, n_embers)
    evy = rng.uniform(22.0, 62.0, n_embers)
    evx = rng.uniform(-14.0, 18.0, n_embers)
    eradius = rng.uniform(1.6, 3.8, n_embers)
    ephase = rng.uniform(0.0, 2 * math.pi, n_embers)

    # Precompute drifting mist texture tile
    mist_small = rng.normal(0.5, 0.25, (height // 16, width // 16)).clip(0.0, 1.0)
    mist_img = (
        Image.fromarray((mist_small * 255).astype(np.uint8))
        .resize((width * 2, height), Image.Resampling.BICUBIC)
        .filter(ImageFilter.GaussianBlur(18))
    )
    mist_tile = np.asarray(mist_img, dtype=np.float32) / 255.0
    # Weight mist heavier toward upper sky and lower ground edges
    y_grad = np.linspace(0.75, 0.45, height)[:, None]
    is_urgent = shot.shot_id == "shot_005"

    out_clip.parent.mkdir(parents=True, exist_ok=True)
    exe = get_ffmpeg_exe()
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "pcm_s16le",
        "-shortest",
        str(out_clip),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    yy, xx = np.mgrid[0:height, 0:width]

    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        t = i / float(fps)

        # Camera path: smooth upward tilt for shot_001, urgent push-in + subtle handheld sway for shot_005
        if is_urgent:
            zoom = 1.04 + 0.14 * (u ** 0.9)
            cx = up_w * 0.50 + math.sin(t * 2.6) * 5.5
            cy = up_h * 0.48 + math.cos(t * 1.9) * 3.5
        else:
            zoom = 1.05 + 0.08 * u
            cx = up_w * 0.50 + math.sin(t * 1.1) * 4.0
            cy = up_h * (0.54 - 0.06 * u)

        crop_w = up_w / zoom
        crop_h = up_h / zoom
        x0 = max(0.0, min(up_w - crop_w, cx - crop_w / 2.0))
        y0 = max(0.0, min(up_h - crop_h, cy - crop_h / 2.0))

        frame_pil = base_img.transform(
            (width, height),
            Image.Transform.EXTENT,
            (x0, y0, x0 + crop_w, y0 + crop_h),
            resample=Image.Resampling.BICUBIC,
        )
        # Mild glow pulse matched to crop
        glow_pil = glow_mask.transform(
            (width, height),
            Image.Transform.EXTENT,
            (x0, y0, x0 + crop_w, y0 + crop_h),
            resample=Image.Resampling.BILINEAR,
        )
        frame = np.asarray(frame_pil, dtype=np.float32)
        g_local = np.asarray(glow_pil, dtype=np.float32)[:, :, None] / 255.0

        # Subtle crimson lightning / energy pulse on red highlights
        pulse = 0.06 + 0.08 * (0.5 + 0.5 * math.sin(t * 2.8 + 0.4 * math.sin(t * 6.5)))
        frame[:, :, 0] += g_local[:, :, 0] * 38.0 * pulse
        frame[:, :, 1] += g_local[:, :, 0] * 6.0 * pulse

        # Drifting crimson-tinted nocturnal mist
        shift = int((t * (28.0 if is_urgent else 18.0)) % width)
        mist_slice = mist_tile[:, shift : shift + width] * y_grad
        frame[:, :, 0] += mist_slice * 16.0
        frame[:, :, 1] += mist_slice * 5.0
        frame[:, :, 2] += mist_slice * 8.0

        # Glowing drifting embers
        ember_layer = np.zeros((height, width), dtype=np.float32)
        for k in range(n_embers):
            ex = (ex0[k] + evx[k] * t + 8.0 * math.sin(t * 2.0 + ephase[k])) % width
            ey = (ey0[k] - evy[k] * t) % height
            r = eradius[k]
            bx0, bx1 = max(0, int(ex - r * 2.5)), min(width, int(ex + r * 2.5 + 1))
            by0, by1 = max(0, int(ey - r * 2.5)), min(height, int(ey + r * 2.5 + 1))
            if bx1 > bx0 and by1 > by0:
                dist2 = (xx[by0:by1, bx0:bx1] - ex) ** 2 + (yy[by0:by1, bx0:bx1] - ey) ** 2
                flicker = 0.55 + 0.45 * math.sin(t * 7.0 + ephase[k])
                ember_layer[by0:by1, bx0:bx1] += flicker * np.exp(-dist2 / (r * r))

        frame[:, :, 0] += ember_layer * 175.0
        frame[:, :, 1] += ember_layer * 55.0
        frame[:, :, 2] += ember_layer * 28.0

        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())

    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"2.5D motion ffmpeg pipe failed ({ret}): {stderr}")
    return count_video_frames(out_clip)


def _normalize_external_video_ai_clip(
    raw_ai_clip: Path,
    audio_path: Path,
    out_clip: Path,
    shot: Shot,
    width: int,
    height: int,
    fps: int,
) -> int:
    """Conform a Colab Wan 2.2 / LTX-2 clip to exact shot frame length & 24fps grid."""
    target_frames = max(1, int(round((shot.duration_sec or 4.0) * fps)))
    raw_dur = probe_media_duration(raw_ai_clip)
    target_dur = target_frames / float(fps)
    speed_factor = target_dur / max(0.2, raw_dur)
    # Clamp gentle time-stretch or loop if raw clip is much shorter
    vf = (
        f"setpts={speed_factor:.6f}*PTS,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )
    run_ffmpeg(
        [
            "-i",
            str(raw_ai_clip),
            "-i",
            str(audio_path),
            "-vf",
            vf,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(target_frames),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(out_clip),
        ]
    )
    return count_video_frames(out_clip)


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape Windows path characters for FFmpeg libass filter syntax."""
    s = path.resolve().as_posix()
    return s.replace(":", r"\:")


def _sync_missing_audio_metadata(base_dir: Path, episode: EpisodeScript) -> None:
    """Hydrate audio_path/speech_duration_sec/duration_sec from generated assets if unset."""
    for shot in episode.shots:
        wav_rel = f"assets/audio/{shot.shot_id}.wav"
        wav_abs = base_dir / wav_rel
        mp3_abs = base_dir / "assets" / "audio" / f"{shot.shot_id}_raw.mp3"
        if not shot.audio_path and wav_abs.exists():
            shot.audio_path = wav_rel
        if shot.speech_duration_sec is None and mp3_abs.exists():
            shot.speech_duration_sec = round(probe_mp3_duration(mp3_abs), 4)
        if shot.duration_sec is None and wav_abs.exists():
            shot.duration_sec = round(probe_media_duration(wav_abs), 6)


def render_pilot_video(base_dir: Path, script_path: Path, reuse_clips: bool = False) -> Path:
    """Execute full Step 6 local assembly -> output/pilot_30s.mp4."""
    episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    _sync_missing_audio_metadata(base_dir, episode)
    prep_dir = base_dir / "assets" / "keyframes_16x9"
    clips_dir = base_dir / "assets" / "clips"
    out_dir = base_dir / "output"
    bgm_path = base_dir / "assets" / "bgm" / "pilot_bgm.wav"
    sfx_path = base_dir / "assets" / "sfx" / "pilot_sfx.wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    cues: list[tuple[float, str, float]] = []
    current_t = 0.0
    rendered_clips: list[Path] = []

    for idx, shot in enumerate(episode.shots):
        if not shot.keyframe_path or not shot.audio_path or shot.duration_sec is None:
            raise RuntimeError(f"{shot.shot_id} incomplete; ensure Step 3 and Step 4 finished.")

        src_img = base_dir / shot.keyframe_path
        norm_img = prep_dir / f"{shot.shot_id}.png"
        _prepare_16x9_image(src_img, norm_img, episode.width, episode.height)

        audio_path = base_dir / shot.audio_path
        clip_path = clips_dir / f"{shot.shot_id}.mp4"
        ai_override = base_dir / "assets" / "video_ai" / f"{shot.shot_id}.mp4"

        if reuse_clips and clip_path.exists():
            n_frames = count_video_frames(clip_path)
            shot.render_mode_used = shot.render_mode_used or (
                "local_2.5d_motion_sim" if shot.shot_type.is_video_ai else shot.shot_type.value
            )
        elif shot.shot_type.is_video_ai and ai_override.exists():
            n_frames = _normalize_external_video_ai_clip(
                ai_override, audio_path, clip_path, shot, episode.width, episode.height, episode.fps
            )
            shot.render_mode_used = "video_ai_external"
        elif shot.shot_type.is_video_ai:
            n_frames = _render_local_25d_motion_clip(
                norm_img, audio_path, clip_path, shot, episode.width, episode.height, episode.fps
            )
            shot.render_mode_used = "local_2.5d_motion_sim"
        else:
            n_frames = _render_ken_burns_clip(
                norm_img, audio_path, clip_path, shot, episode.width, episode.height, episode.fps
            )
            shot.render_mode_used = shot.shot_type.value

        actual_dur = n_frames / float(episode.fps)
        shot.video_clip_path = f"assets/clips/{shot.shot_id}.mp4"
        shot.duration_sec = round(actual_dur, 6)
        rendered_clips.append(clip_path)

        cues.append((current_t, shot.sfx_cue or "none", actual_dur))
        current_t += actual_dur
        print(
            f"[{idx+1}/{len(episode.shots)}] Rendered {shot.shot_id} ({shot.render_mode_used}) "
            f"-> {n_frames} frames ({actual_dur:.3f}s)"
        )

    script_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")

    # 1. Concatenate all shot clips with sample-accurate concat filter
    raw_concat = out_dir / "pilot_raw.mp4"
    concat_inputs: list[str] = []
    streams: list[str] = []
    for i, clip in enumerate(rendered_clips):
        concat_inputs.extend(["-i", str(clip)])
        streams.append(f"[{i}:v][{i}:a]")
    fade_out_start = max(0.0, current_t - 0.8)
    filter_complex = (
        f"{''.join(streams)}concat=n={len(rendered_clips)}:v=1:a=1[vcat][acat];"
        f"[vcat]fade=t=in:st=0:d=0.45,fade=t=out:st={fade_out_start:.3f}:d=0.80[vout]"
    )
    run_ffmpeg(
        [
            *concat_inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[acat]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            str(raw_concat),
        ]
    )

    # 2. Generate BGM & SFX matched to exact raw_concat timeline
    total_dur = probe_media_duration(raw_concat)
    generate_cinematic_bgm(bgm_path, total_dur)
    build_sfx_track(sfx_path, cues, total_dur)

    # 3. Mix audio with automatic sidechain ducking on BGM under dialogue
    mixed_mp4 = out_dir / "pilot_mixed.mp4"
    run_ffmpeg(
        [
            "-i",
            str(raw_concat),
            "-i",
            str(bgm_path),
            "-i",
            str(sfx_path),
            "-filter_complex",
            (
                "[0:a]asplit=2[voice_sc][voice_mix];"
                "[1:a]volume=0.55[bgm_gain];"
                "[bgm_gain][voice_sc]sidechaincompress="
                "threshold=0.015:ratio=6:attack=45:release=380:makeup=1[bgm_ducked];"
                "[2:a]volume=0.85[sfx_gain];"
                "[voice_mix][bgm_ducked][sfx_gain]amix=inputs=3:duration=first:normalize=0,"
                "alimiter=limit=0.95[aout]"
            ),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(mixed_mp4),
        ]
    )

    # 4. Build global-offset .ass & .srt subtitles and burn hardsubs onto pilot_30s.mp4
    ass_path = out_dir / "episode_01.ass"
    srt_path = out_dir / "episode_01.srt"
    build_subtitles(base_dir, script_path, ass_path, srt_path, episode=episode)

    final_mp4 = out_dir / "pilot_30s.mp4"
    escaped_ass = _escape_ffmpeg_filter_path(ass_path)
    run_ffmpeg(
        [
            "-i",
            str(mixed_mp4),
            "-vf",
            f"ass='{escaped_ass}'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(final_mp4),
        ]
    )
    print(f"FINAL PILOT VIDEO READY: {final_mp4} ({total_dur:.2f}s)")
    return final_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pilot_30s.mp4 locally")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "script.json",
    )
    parser.add_argument(
        "--reuse-clips",
        action="store_true",
        help="Reuse existing rendered shot clips in assets/clips/",
    )
    args = parser.parse_args()
    render_pilot_video(args.script.resolve().parent, args.script.resolve(), reuse_clips=args.reuse_clips)


if __name__ == "__main__":
    main()
