import os
import json
import wave
import shutil
import argparse
import asyncio
import subprocess
import urllib.parse
from pathlib import Path
import requests
import numpy as np
import edge_tts
import pysubs2
import imageio_ffmpeg
from gtts import gTTS
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont

FPS = 24
ROOT_DIR = Path(".")
ASSETS_DIR = ROOT_DIR / "assets"
CHAR_DIR = ASSETS_DIR / "characters"
HUD_DIR = ASSETS_DIR / "ui_hud"
BG_DIR = ASSETS_DIR / "backgrounds"
STOCK_DIR = ASSETS_DIR / "stock_clips"
SFX_DIR = ASSETS_DIR / "sfx"
BUILD_DIR = ROOT_DIR / "build_output"

FFMPEG_EXE = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
VILLAIN_IDS = {"mac_khieu", "quai_vat_tuan", "phu_giap"}


def get_dims(mode: str) -> tuple[int, int]:
    return (1080, 1920) if mode == "tiktok" else (1920, 1080)


def normalize_state(state: str) -> str:
    mapping = {
        "armor_base": "armored",
        "henshin": "awakened",
    }
    return mapping.get(state, state or "civilian")


def resolve_scene_chars(scene: dict):
    """Hỗ trợ cả kịch bản 2 nhân vật (left_char/right_char), kịch bản hiện đại (speaker/attacker) và 1 nhân vật (char_id/position)."""
    if "left_char" in scene or "right_char" in scene:
        left_cfg = scene.get("left_char")
        right_cfg = scene.get("right_char")
        if left_cfg:
            left_cfg = {**left_cfg, "state": normalize_state(left_cfg.get("state", "civilian"))}
        if right_cfg:
            right_cfg = {**right_cfg, "state": normalize_state(right_cfg.get("state", "civilian"))}
        speaker_side = scene.get("speaker", "left")
        spk_cfg = left_cfg if speaker_side == "left" else right_cfg
        return left_cfg, right_cfg, None, speaker_side, spk_cfg["id"]

    char_id = scene.get("char_id") or scene.get("speaker", "hoa_do")
    state = normalize_state(scene.get("state", "civilian"))
    pos = scene.get("position", "center")
    cfg = {"id": char_id, "state": state}
    left_cfg = cfg if pos == "left" else None
    right_cfg = cfg if pos == "right" else None
    center_cfg = cfg if pos == "center" else None
    return left_cfg, right_cfg, center_cfg, pos, char_id


def resolve_hud_type(scene: dict, act_id: str) -> str:
    if "hud" in scene:
        return scene["hud"]
    if scene.get("show_hud", False):
        if act_id == "act_06" or scene.get("bg_id") == "bg_mecha_sky":
            return "cockpit"
        return "aegis"
    return "none"


def resolve_shot_type(scene: dict) -> str:
    explicit = scene.get("shot_type")
    if explicit:
        if explicit == "stock_clip":
            clip_name = scene.get("clip_name", "")
            if clip_name and (STOCK_DIR / clip_name).exists():
                return "stock_clip"
            return "eye_cutin"
        return explicit

    if "left_char" in scene or "right_char" in scene:
        return "dual_stage"
    if scene.get("cut_in", False) or scene.get("state") in ("henshin", "awakened"):
        return "eye_cutin"
    return "portrait_drama"


def resolve_voice_filter(scene: dict, speaker_id: str) -> str:
    if "voice_filter" in scene:
        return scene["voice_filter"]
    if speaker_id in VILLAIN_IDS:
        return "villain"
    if normalize_state(scene.get("state", "civilian")) != "civilian":
        return "helmet"
    return "none"


def resolve_sfx(scene: dict, act_id: str):
    if "sfx" in scene:
        return scene["sfx"]
    if scene.get("state") == "henshin":
        return "henshin"
    if scene.get("flash_start", False):
        return "power_charge"
    if scene.get("camera") == "shake" and act_id in ("act_02", "act_03", "act_05", "act_06"):
        return "explosion"
    return None


# ==============================================================================
# 1. XỬ LÝ GIỌNG LỒNG TIẾNG TOKUSATSU (MŨ BẢO HIỂM / PHẢN DIỆN / BỘ ĐÀM)
# ==============================================================================
def process_tokusatsu_voice(raw_mp3: Path, final_wav: Path, voice_filter: str = "none"):
    """Chuyển MP3 sang WAV 48kHz và áp dụng bộ lọc âm thanh đặc trưng Super Sentai."""
    if final_wav.exists() and final_wav.stat().st_size > 1000:
        return

    if voice_filter == "helmet":
        af = "highpass=f=300,lowpass=f=3500,acompressor=threshold=-18dB:ratio=4,aecho=0.8:0.88:12:0.25"
    elif voice_filter == "villain":
        af = "asetrate=24000*0.86,aresample=48000,aecho=0.8:0.85:35:0.35,acompressor"
    elif voice_filter == "radio":
        af = "highpass=f=450,lowpass=f=3000,acompressor=threshold=-20dB:ratio=5"
    else:
        af = "anull"

    subprocess.run(
        [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", str(raw_mp3), "-af", af, "-ac", "1", "-ar", "48000", str(final_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def get_rms_envelopes(wav_path: Path, fps: int = FPS) -> np.ndarray:
    """Đo biên độ âm lượng từng frame để điều khiển nhép miệng hoặc phát sáng Hào quang Nguyên Giáp."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    spf = max(1, int(sr / fps))
    n_frames = int(np.ceil(len(samples) / spf))
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = samples[i * spf : (i + 1) * spf]
        if len(seg) > 0:
            rms[i] = np.sqrt(np.mean(seg ** 2))

    peak = np.max(rms) if np.max(rms) > 0 else 1.0
    return rms / peak


async def generate_voice_and_ass(
    text: str,
    char_cfg: dict,
    voice_filter: str,
    raw_mp3: Path,
    proc_wav: Path,
    ass_path: Path,
    mode: str,
    sem: asyncio.Semaphore
) -> float:
    """Sinh giọng đọc (hoặc giữ nguyên file RVC/thu âm có sẵn) và xuất phụ đề .ass."""
    async with sem:
        if not raw_mp3.exists() or raw_mp3.stat().st_size < 500:
            voice = char_cfg.get("voice", "vi-VN-NamMinhNeural")
            attempts = [
                {"voice": voice, "rate": char_cfg.get("rate", "+0%"), "pitch": char_cfg.get("pitch", "+0Hz")},
                {"voice": voice},
            ]
            success = False
            for idx, kwargs in enumerate(attempts):
                try:
                    comm = edge_tts.Communicate(text=text, **kwargs)
                    audio_bytes = bytearray()
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            audio_bytes.extend(chunk["data"])
                    if len(audio_bytes) >= 500:
                        with open(raw_mp3, "wb") as f:
                            f.write(audio_bytes)
                        success = True
                        break
                except Exception:
                    await asyncio.sleep(0.4 + idx * 0.4)
            if not success:
                print(f"  [TTS Fallback -> gTTS] {raw_mp3.name}")
                await asyncio.to_thread(lambda: gTTS(text=text, lang="vi").save(str(raw_mp3)))
            await asyncio.sleep(0.15)

        process_tokusatsu_voice(raw_mp3, proc_wav, voice_filter=voice_filter)
        duration = MP3(str(raw_mp3)).info.length
        w, h = get_dims(mode)

        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = str(w)
        subs.info["PlayResY"] = str(h)

        r, g, b = char_cfg.get("sub_color", [255, 255, 255])
        name_tag = char_cfg.get("display_name", "")
        prefix = f"【{name_tag}】 " if name_tag and name_tag != "NGƯỜI DẪN CHUYỆN" else ""

        style = pysubs2.SSAStyle(
            fontname="Arial",
            fontsize=54 if mode == "tiktok" else 46,
            bold=True,
            primarycolor=pysubs2.Color(r, g, b, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 0),
            backcolor=pysubs2.Color(0, 0, 0, 165),
            outline=3.5,
            shadow=2.0,
            alignment=2,
            marginv=240 if mode == "tiktok" else 65,
            marginl=80,
            marginr=80
        )
        subs.styles["Default"] = style
        subs.events.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=0.05),
                end=pysubs2.make_time(s=duration + 0.35),
                text=prefix + text
            )
        )
        subs.save(str(ass_path))
        return duration


# ==============================================================================
# 2. HIỆU ỨNG TOKUSATSU HIỆN ĐẠI (IMPACT FRAME, EYE CUT-IN, LOWER-THIRD)
# ==============================================================================
def apply_impact_frame(canvas: Image.Image, f_idx: int) -> Image.Image:
    """Chớp 3 khung hình va chạm trắng -> đảo màu âm bản -> đen trắng tương phản cao."""
    if f_idx == 0:
        return Image.new("RGBA", canvas.size, (255, 255, 255, 240))
    if f_idx == 1:
        return ImageOps.invert(canvas.convert("RGB")).convert("RGBA")
    if f_idx == 2:
        gray = ImageOps.grayscale(canvas).point(lambda p: 255 if p > 125 else 0)
        return gray.convert("RGBA")
    return canvas


def draw_eye_cutin_banner(canvas: Image.Image, char_id: str, theme_rgb: tuple, progress: float) -> Image.Image:
    """Dải băng phóng to ánh mắt chiến binh trượt ngang màn hình khi Điểm danh / Tung chiêu."""
    if progress > 0.55:
        return canvas
    w, h = canvas.size
    band_h = int(h * 0.20)
    y_mid = int(h * 0.38)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = theme_rgb

    slide_x = int((1.0 - min(1.0, progress * 3.5)) * w)
    draw.rectangle([slide_x, y_mid - band_h // 2, w + slide_x, y_mid + band_h // 2], fill=(r, g, b, 210))
    draw.line([(0, y_mid - band_h // 2), (w, y_mid - band_h // 2)], fill=(255, 255, 255, 255), width=6)
    draw.line([(0, y_mid + band_h // 2), (w, y_mid + band_h // 2)], fill=(255, 255, 255, 255), width=6)

    eye_file = CHAR_DIR / char_id / "eye_strip.png"
    if eye_file.exists():
        eye_im = Image.open(eye_file).convert("RGBA").resize((int(w * 0.60), band_h - 12), Image.Resampling.LANCZOS)
        overlay.alpha_composite(eye_im, dest=(int(w * 0.20) + slide_x, y_mid - band_h // 2 + 6))

    return Image.alpha_composite(canvas, overlay)


def draw_lower_third_tag(canvas: Image.Image, display_name: str, role_text: str, theme_rgb: tuple, mode: str):
    """Vẽ bảng tên nhân vật hiện đại ở góc dưới bên trái trong các cảnh diễn xuất."""
    if not display_name or display_name == "NGƯỜI DẪN CHUYỆN":
        return
    w, h = canvas.size
    box_w, box_h = (520, 86) if mode == "youtube" else (580, 92)
    x0, y0 = (45, h - 220) if mode == "youtube" else (40, h - 420)

    tag_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tag_layer)
    r, g, b = theme_rgb
    d.polygon([(x0, y0), (x0 + box_w, y0), (x0 + box_w - 30, y0 + box_h), (x0, y0 + box_h)], fill=(15, 20, 32, 215))
    d.rectangle([x0, y0, x0 + 12, y0 + box_h], fill=(r, g, b, 255))

    try:
        f_main = ImageFont.truetype("arialbd.ttf", 26)
        f_sub = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        f_main = ImageFont.load_default()
        f_sub = ImageFont.load_default()

    d.text((x0 + 26, y0 + 12), display_name, fill=(255, 255, 255, 255), font=f_main)
    d.text((x0 + 26, y0 + 48), role_text, fill=(min(255, r + 90), min(255, g + 90), min(255, b + 90), 240), font=f_sub)
    canvas.alpha_composite(tag_layer)


# ==============================================================================
# 3. BỐI CẢNH CỐ ĐỊNH & SPRITE ĐA LỚP / AI ART
# ==============================================================================
def ensure_locked_background(bg_id: str, bg_info: dict, style_suffix: str, mode: str) -> Path:
    w, h = get_dims(mode)
    out_jpg = BG_DIR / f"{bg_id}_{mode}.jpg"
    if out_jpg.exists() and out_jpg.stat().st_size > 5000:
        return out_jpg

    full_prompt = f"{bg_info['prompt']}, {style_suffix}"
    seed = bg_info.get("seed", 1000)
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(full_prompt)}?width={w}&height={h}&seed={seed}&nologo=true&model=flux"
    print(f"  [Locked BG] Đang khởi tạo bối cảnh cố định '{bg_id}' (Seed={seed})...")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 5000:
            with open(out_jpg, "wb") as f:
                f.write(resp.content)
            return out_jpg
    except Exception as e:
        print(f"   [BG Fallback] Chuyển sang vẽ bối cảnh nội bộ cho '{bg_id}': {e}")

    fb_colors = bg_info.get("fallback_colors")
    c_top = fb_colors[0] if fb_colors and len(fb_colors) >= 2 else [18, 24, 45]
    c_bot = fb_colors[1] if fb_colors and len(fb_colors) >= 2 else [58, 74, 130]

    img = Image.new("RGB", (w, h), tuple(c_top))
    draw = ImageDraw.Draw(img)
    for y in range(0, h, 4):
        t = y / h
        c_r = int(c_top[0] + (c_bot[0] - c_top[0]) * t)
        c_g = int(c_top[1] + (c_bot[1] - c_top[1]) * t)
        c_b = int(c_top[2] + (c_bot[2] - c_top[2]) * t)
        draw.rectangle([0, y, w, y + 4], fill=(c_r, c_g, c_b))
    img.save(out_jpg, quality=92)
    return out_jpg


class CharacterSpriteSlot:
    """Ưu tiên nạp ảnh AI chất lượng cao (<state>.png) nếu có; mặc định dùng bộ sprite 4 trạng thái đa lớp."""
    def __init__(self, char_id: str, state: str, cw: int, ch: int, flip_horizontal: bool = False):
        self.state = state
        self.custom_art = None
        c_dir = CHAR_DIR / char_id

        def load_img(fname: str) -> Image.Image:
            im = Image.open(c_dir / fname).convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
            return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip_horizontal else im

        ai_override = c_dir / f"{state}.png"
        if ai_override.exists():
            self.custom_art = load_img(f"{state}.png")
            return

        if state == "civilian":
            self.base_open = load_img("civ_open.png")
            self.base_blink = load_img("civ_blink.png")
            self.m_closed = load_img("mouth_closed.png")
            self.m_small = load_img("mouth_small.png")
            self.m_wide = load_img("mouth_wide.png")
        elif state == "broken_helmet":
            self.base_open = load_img("broken_open.png")
            self.base_blink = load_img("broken_blink.png")
            self.m_closed = load_img("mouth_closed.png")
            self.m_small = load_img("mouth_small.png")
            self.m_wide = load_img("mouth_wide.png")
            self.aura = load_img("awake_aura.png")
        elif state == "awakened":
            self.base_open = load_img("awake_base.png")
            self.visor_glow = load_img("awake_visor_glow.png")
            self.aura = load_img("awake_aura.png")
        else:  # armored
            self.base_open = load_img("armor_base.png")
            self.visor_glow = load_img("armor_visor_glow.png")
            self.aura = load_img("armor_aura.png")

    def render_to_canvas(self, canvas: Image.Image, pos: tuple[int, int], frame_idx: int, vol: float, is_speaking: bool):
        if self.custom_art is not None:
            temp = self.custom_art.copy()
        else:
            temp = Image.new("RGBA", self.base_open.size, (0, 0, 0, 0))
            if self.state in ("civilian", "broken_helmet"):
                if self.state == "broken_helmet":
                    temp.alpha_composite(self.aura)
                is_blink = (frame_idx % (FPS * 3 + 5)) in (0, 1, 2)
                temp.alpha_composite(self.base_blink if is_blink else self.base_open)
                eff_vol = vol if is_speaking else 0.0
                mouth = self.m_closed if eff_vol < 0.12 else (self.m_small if eff_vol < 0.45 else self.m_wide)
                temp.alpha_composite(mouth)
            else:
                if is_speaking and vol > 0.10:
                    alpha_scale = min(1.0, vol * 1.4)
                    dyn_aura = self.aura.copy()
                    _, _, _, a = dyn_aura.split()
                    dyn_aura.putalpha(a.point(lambda p: int(p * alpha_scale)))
                    temp.alpha_composite(dyn_aura)
                temp.alpha_composite(self.base_open)
                if is_speaking and vol > 0.12:
                    temp.alpha_composite(self.visor_glow)

        if not is_speaking:
            rgb = ImageEnhance.Brightness(temp.convert("RGB")).enhance(0.72)
            r, g, b = rgb.split()
            temp = Image.merge("RGBA", (r, g, b, temp.split()[3]))

        canvas.alpha_composite(temp, dest=pos)


# ==============================================================================
# 4. RENDER PHÂN CẢNH 6 GÓC MÁY & TRỘN SFX / BGM THÍCH ỨNG
# ==============================================================================
def mix_sfx_and_finalize(raw_mp4: Path, sfx_name: str, final_mp4: Path):
    """Ghép tiếng động chiến đấu (explosion.wav, henshin.wav, sword_slash.wav...) vào phân cảnh."""
    sfx_path = SFX_DIR / f"{sfx_name}.wav" if sfx_name else None
    if sfx_path and sfx_path.exists():
        subprocess.run([
            FFMPEG_EXE, "-y", "-loglevel", "error",
            "-i", str(raw_mp4),
            "-i", str(sfx_path),
            "-filter_complex", "[1:a]volume=0.65[sfx];[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=1[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            str(final_mp4)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_mp4.unlink(missing_ok=True)
    else:
        raw_mp4.replace(final_mp4)


def render_sentai_scene(
    scene: dict,
    act_id: str,
    chars_reg: dict,
    bg_jpg: Path,
    proc_wav: Path,
    ass_path: Path,
    out_mp4: Path,
    mode: str,
    force: bool = False
):
    """Render phân cảnh linh hoạt theo 6 shot_type hiện đại + tương thích ngược kịch bản 2 nhân vật."""
    if not force and out_mp4.exists() and out_mp4.stat().st_size > 10000:
        return

    w, h = get_dims(mode)
    shot_type = resolve_shot_type(scene)
    envelopes = get_rms_envelopes(proc_wav, fps=FPS)
    total_frames = len(envelopes) + int(FPS * 0.4)
    total_dur = round(total_frames / FPS, 2)
    escaped_ass = ass_path.resolve().as_posix().replace(":", r"\:")
    sfx_name = resolve_sfx(scene, act_id)

    # TRƯỜNG HỢP ĐẶC BIỆT: CHÈN STOCK FOOTAGE (.MP4) HOẶC LIVEPORTRAIT CLIP
    live_clip = None
    if shot_type == "portrait_drama":
        cand = scene.get("video_clip")
        if cand and Path(cand).exists():
            live_clip = Path(cand)
        else:
            _, _, _, _, spk_id_tmp = resolve_scene_chars(scene)
            lp_cand = CHAR_DIR / spk_id_tmp / f"liveportrait_{normalize_state(scene.get('state', 'civilian'))}.mp4"
            if lp_cand.exists():
                live_clip = lp_cand

    if shot_type == "stock_clip" or live_clip is not None:
        clip_src = live_clip if live_clip is not None else (STOCK_DIR / scene.get("clip_name", ""))
        raw_scene = out_mp4.with_suffix(".raw.mp4")
        subprocess.run([
            FFMPEG_EXE, "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip_src),
            "-i", str(proc_wav),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},ass='{escaped_ass}',format=yuv420p",
            "-af", "apad=pad_dur=0.4",
            "-t", str(total_dur),
            "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            str(raw_scene)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mix_sfx_and_finalize(raw_scene, sfx_name, out_mp4)
        return

    scale_w, scale_h = int(w * 1.12), int(h * 1.12)
    bg_raw = Image.open(bg_jpg).convert("RGBA").resize((scale_w, scale_h), Image.Resampling.LANCZOS)
    max_dx, max_dy = scale_w - w, scale_h - h

    c_scale = 0.78 if mode == "youtube" else 0.72
    cw, ch = int(600 * c_scale), int(750 * c_scale)

    left_cfg, right_cfg, center_cfg, speaker_side, speaker_id = resolve_scene_chars(scene)
    spk_cfg = chars_reg.get(speaker_id, {"display_name": "", "role": "", "primary_rgb": [255, 200, 80]})
    theme_rgb = tuple(spk_cfg.get("primary_rgb", [255, 200, 80]))
    camera = scene.get("camera", "zoom_in")
    hud_type = resolve_hud_type(scene, act_id)

    left_slot = right_slot = center_slot = None
    split_slots = []

    if shot_type == "dual_stage":
        left_slot = CharacterSpriteSlot(left_cfg["id"], left_cfg["state"], cw, ch, flip_horizontal=False) if left_cfg else None
        right_slot = CharacterSpriteSlot(right_cfg["id"], right_cfg["state"], cw, ch, flip_horizontal=True) if right_cfg else None
    elif shot_type == "action_clash":
        atk_id = scene.get("attacker", speaker_id)
        def_id = scene.get("defender", "quai_vat_tuan")
        atk_state = normalize_state(scene.get("attacker_state", scene.get("state", "armored")))
        def_state = normalize_state(scene.get("defender_state", "armored"))
        left_slot = CharacterSpriteSlot(atk_id, atk_state, cw, ch, flip_horizontal=False)
        right_slot = CharacterSpriteSlot(def_id, def_state, cw, ch, flip_horizontal=True)
    elif shot_type == "com_link":
        panel_ids = scene.get("panel_chars", ["hoa_do", "thuy_lam", "phong_luc"])
        st = normalize_state(scene.get("state", "civilian"))
        pw = w // len(panel_ids)
        p_cw = min(cw, int(pw * 0.92))
        p_ch = int(p_cw * 750 / 600)
        for pid in panel_ids:
            split_slots.append((pid, CharacterSpriteSlot(pid, st, p_cw, p_ch)))
    elif shot_type in ("portrait_drama", "eye_cutin"):
        if left_cfg:
            left_slot = CharacterSpriteSlot(left_cfg["id"], left_cfg["state"], cw, ch, flip_horizontal=False)
        if right_cfg:
            right_slot = CharacterSpriteSlot(right_cfg["id"], right_cfg["state"], cw, ch, flip_horizontal=True)
        if center_cfg:
            center_slot = CharacterSpriteSlot(center_cfg["id"], center_cfg["state"], cw, ch, flip_horizontal=False)

    hud_overlay = None
    suffix_mode = "16x9" if mode == "youtube" else "9x16"
    if hud_type in ("aegis", "cockpit"):
        hud_file = HUD_DIR / f"hud_{hud_type}_{suffix_mode}.png"
        if hud_file.exists():
            hud_overlay = Image.open(hud_file).convert("RGBA")

    raw_scene = out_mp4.with_suffix(".raw.mp4")
    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgba", "-r", str(FPS),
        "-i", "-",
        "-i", str(proc_wav),
        "-vf", f"ass='{escaped_ass}',format=yuv420p",
        "-af", "apad=pad_dur=0.4",
        "-t", str(total_dur),
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        str(raw_scene)
    ]

    ffmpeg_log = out_mp4.with_suffix(".ffmpeg.log")
    with open(ffmpeg_log, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f)

        if mode == "youtube":
            left_x, right_x = 110, w - cw - 110
            center_x = (w - cw) // 2
            base_y = h - ch - 25
        else:
            left_x, right_x = 30, w - cw - 30
            center_x = (w - cw) // 2
            base_y = h - ch - 240

        for idx in range(total_frames):
            prog = idx / max(1, total_frames - 1)
            vol = envelopes[idx] if idx < len(envelopes) else 0.0

            if camera == "shake" or (shot_type == "action_clash" and idx < 10):
                amp = 20 if (idx < FPS * 0.6 or vol > 0.5) else 4
                sx = int(np.random.uniform(-amp, amp))
                sy = int(np.random.uniform(-amp, amp))
                cx = int(np.clip(max_dx // 2 + sx, 0, max_dx))
                cy = int(np.clip(max_dy // 2 + sy, 0, max_dy))
            elif camera == "crash_zoom":
                fast_p = min(1.0, idx / (FPS * 0.45))
                cx, cy = int(max_dx * fast_p * 0.5), int(max_dy * fast_p * 0.5)
            elif camera == "pan_right":
                cx, cy = int(max_dx * prog), max_dy // 2
            elif camera == "pan_left":
                cx, cy = int(max_dx * (1.0 - prog)), max_dy // 2
            elif camera == "zoom_out":
                cx, cy = int(max_dx * (1.0 - prog) * 0.5), int(max_dy * (1.0 - prog) * 0.5)
            else:
                cx, cy = int(max_dx * prog * 0.5), int(max_dy * prog * 0.5)

            canvas = bg_raw.crop((cx, cy, cx + w, cy + h)).copy()
            breath = int(np.sin(idx * 0.15) * 3)

            if shot_type == "action_clash" and left_slot and right_slot:
                dash_p = min(1.0, idx / (FPS * 0.35))
                atk_x = int(60 + dash_p * (w * 0.24))
                def_x = int(w - cw - 80 + (18 if idx >= 6 else 0))
                right_slot.render_to_canvas(canvas, (def_x, base_y), idx, vol, is_speaking=False)
                left_slot.render_to_canvas(canvas, (atk_x, base_y), idx, vol, is_speaking=True)
                if scene.get("impact_frame", True) and idx in (6, 7, 8):
                    canvas = apply_impact_frame(canvas, idx - 6)
            elif shot_type == "com_link" and split_slots:
                panel_w = w // len(split_slots)
                for p_idx, (pid, slot) in enumerate(split_slots):
                    px0 = p_idx * panel_w
                    sub_bg = canvas.crop((px0, 0, px0 + panel_w, h))
                    is_spk = (pid == speaker_id)
                    if not is_spk:
                        sub_bg = ImageEnhance.Brightness(sub_bg.convert("RGB")).enhance(0.58).convert("RGBA")
                    slot_w, slot_h = slot.base_open.size if slot.custom_art is None else slot.custom_art.size
                    slot.render_to_canvas(sub_bg, ((panel_w - slot_w) // 2, h - slot_h - (25 if mode == "youtube" else 240)), idx, vol, is_speaking=is_spk)
                    canvas.paste(sub_bg, (px0, 0))
                    if p_idx > 0:
                        ImageDraw.Draw(canvas).line([(px0, 0), (px0, h)], fill=(255, 255, 255, 230), width=6)
            elif shot_type != "establishing":
                if left_slot:
                    left_slot.render_to_canvas(canvas, (left_x, base_y + breath), idx, vol, is_speaking=(speaker_side == "left"))
                if right_slot:
                    right_slot.render_to_canvas(canvas, (right_x, base_y + breath), idx, vol, is_speaking=(speaker_side == "right"))
                if center_slot:
                    center_slot.render_to_canvas(canvas, (center_x, base_y + breath), idx, vol, is_speaking=True)

            if shot_type in ("portrait_drama", "eye_cutin"):
                draw_lower_third_tag(canvas, spk_cfg.get("display_name", ""), spk_cfg.get("role", ""), theme_rgb, mode)
            if shot_type == "eye_cutin":
                canvas = draw_eye_cutin_banner(canvas, speaker_id, theme_rgb, prog)

            if hud_overlay:
                canvas.alpha_composite(hud_overlay)

            if scene.get("flash_start", False) and idx < 4:
                flash_a = int(215 * (1.0 - idx / 4.0))
                canvas.alpha_composite(Image.new("RGBA", (w, h), (255, 255, 255, flash_a)))

            proc.stdin.write(canvas.tobytes())

        proc.stdin.close()
        ret = proc.wait()

    if ret != 0:
        err_txt = ffmpeg_log.read_text(encoding="utf-8", errors="ignore") if ffmpeg_log.exists() else ""
        raise RuntimeError(f"FFmpeg failed for {out_mp4} (code {ret}):\n{err_txt[-1000:]}")
    ffmpeg_log.unlink(missing_ok=True)
    mix_sfx_and_finalize(raw_scene, sfx_name, out_mp4)


def concat_act_with_adaptive_bgm(chunk_files: list[Path], bgm_cfg: dict, out_mp4: Path):
    """Ghép các cảnh trong Hồi và trộn nhạc nền với tỷ lệ nén thích ứng (Adaptive Audio Ducking)."""
    list_txt = out_mp4.with_suffix(".concat.txt")
    raw_concat = out_mp4.with_suffix(".raw.mp4")

    with open(list_txt, "w", encoding="utf-8") as f:
        for p in chunk_files:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    subprocess.run(
        [FFMPEG_EXE, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(raw_concat)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    bgm_path = Path(bgm_cfg["file"]) if bgm_cfg else None
    if bgm_path and bgm_path.exists():
        vol = bgm_cfg.get("vol", 0.25)
        ratio = bgm_cfg.get("duck_ratio", 5)
        audio_filter = (
            f"[1:a]volume={vol},aloop=loop=-1:size=2e+09[bgm];"
            f"[bgm][0:a]sidechaincompress=threshold=0.03:ratio={ratio}:attack=40:release=350[ducked];"
            f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        subprocess.run([
            FFMPEG_EXE, "-y", "-loglevel", "error",
            "-i", str(raw_concat),
            "-i", str(bgm_path),
            "-filter_complex", audio_filter,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            str(out_mp4)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_concat.unlink(missing_ok=True)
    else:
        raw_concat.replace(out_mp4)

    list_txt.unlink(missing_ok=True)


async def build_episode(ep_json_path: str, mode: str, force: bool = False):
    with open(ROOT_DIR / "global_registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)
    with open(ep_json_path, "r", encoding="utf-8") as f:
        ep_data = json.load(f)

    ep_num = ep_data["episode_number"]
    ep_code = f"ep_{ep_num:03d}"
    ep_work_dir = BUILD_DIR / ep_code / mode
    ep_audio_dir = BUILD_DIR / ep_code / "audio"
    ep_work_dir.mkdir(parents=True, exist_ok=True)
    ep_audio_dir.mkdir(parents=True, exist_ok=True)

    chars_reg = registry["characters"]
    bgs_reg = {**registry["backgrounds"], **ep_data.get("custom_backgrounds", {})}
    bgm_reg = registry["bgm_tracks"]
    style_suffix = registry["style_suffix"]

    print(f"\n=== [{ep_code.upper()}] BƯỚC 1: KIỂM TRA BỐI CẢNH CỐ ĐỊNH ===")
    used_bgs = {sc["bg_id"] for act in ep_data["acts"] for sc in act["scenes"]}
    bg_paths = {bid: ensure_locked_background(bid, bgs_reg[bid], style_suffix, mode) for bid in used_bgs}

    print(f"\n=== [{ep_code.upper()}] BƯỚC 2: SINH LỒNG TIẾNG (BỘ LỌC TOKUSATSU) & PHỤ ĐỀ ===")
    sem = asyncio.Semaphore(1)
    tasks = []
    for act in ep_data["acts"]:
        aid = act["act_id"]
        for s_idx, sc in enumerate(act["scenes"]):
            sid = f"{aid}_s{s_idx:03d}"
            _, _, _, _, spk_char_id = resolve_scene_chars(sc)
            char_cfg = chars_reg[spk_char_id]
            v_filter = resolve_voice_filter(sc, spk_char_id)
            tasks.append(
                generate_voice_and_ass(
                    sc["text"], char_cfg, v_filter,
                    ep_audio_dir / f"{sid}.mp3",
                    ep_audio_dir / f"{sid}_{v_filter}.wav",
                    ep_work_dir / f"{sid}.ass",
                    mode, sem
                )
            )
    await asyncio.gather(*tasks)

    print(f"\n=== [{ep_code.upper()}] BƯỚC 3: RENDER 6 GÓC MÁY & TRỘN SFX / BGM ===")
    act_outputs = []
    for act in ep_data["acts"]:
        aid = act["act_id"]
        chunk_files = []
        for s_idx, sc in enumerate(act["scenes"]):
            sid = f"{aid}_s{s_idx:03d}"
            out_chunk = ep_work_dir / f"{sid}.mp4"
            _, _, _, _, spk_char_id = resolve_scene_chars(sc)
            v_filter = resolve_voice_filter(sc, spk_char_id)
            shot_name = resolve_shot_type(sc)
            hud_name = resolve_hud_type(sc, aid)
            print(f"  -> {sid} [Shot={shot_name}, Voice={v_filter}, HUD={hud_name}, Cam={sc.get('camera')}]")
            render_sentai_scene(
                scene=sc,
                act_id=aid,
                chars_reg=chars_reg,
                bg_jpg=bg_paths[sc["bg_id"]],
                proc_wav=ep_audio_dir / f"{sid}_{v_filter}.wav",
                ass_path=ep_work_dir / f"{sid}.ass",
                out_mp4=out_chunk,
                mode=mode,
                force=force
            )
            chunk_files.append(out_chunk)

        act_mp4 = ep_work_dir / f"{ep_code}_{aid}_{mode}.mp4"
        bgm_cfg = bgm_reg.get(act.get("bgm_id", "bgm_daily"))
        concat_act_with_adaptive_bgm(chunk_files, bgm_cfg, act_mp4)
        act_outputs.append(act_mp4)

    final_ep = BUILD_DIR / f"NguHanhChienDoi_{ep_code}_{mode.upper()}_FULL.mp4"
    master_txt = ep_work_dir / "master_concat.txt"
    with open(master_txt, "w", encoding="utf-8") as f:
        for p in act_outputs:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    subprocess.run([
        FFMPEG_EXE, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(master_txt), "-c", "copy", str(final_ep)
    ], check=True)
    print(f"\n[HOÀN TẤT TẬP {ep_num}] File phim đã xuất tại: {final_ep.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engine Sản Xuất 100 Tập Ngũ Hành Chiến Đội (Modern Tokusatsu Edition)")
    parser.add_argument("--ep", default="episodes/ep_001.json", help="Đường dẫn file JSON kịch bản tập phim")
    parser.add_argument("--mode", choices=["youtube", "tiktok"], default="youtube", help="Định dạng xuất phim")
    parser.add_argument("--force", action="store_true", help="Render lại toàn bộ các phân cảnh kể cả khi đã có file cache")
    args = parser.parse_args()

    asyncio.run(build_episode(args.ep, args.mode, force=args.force))
