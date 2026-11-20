import os
import json
import shutil
import wave
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT_DIR = Path(".")
ASSETS_DIR = ROOT_DIR / "assets"
CHAR_DIR = ASSETS_DIR / "characters"
HUD_DIR = ASSETS_DIR / "ui_hud"
BGM_DIR = ASSETS_DIR / "bgm"
BG_DIR = ASSETS_DIR / "backgrounds"
SFX_DIR = ASSETS_DIR / "sfx"
STOCK_DIR = ASSETS_DIR / "stock_clips"
EPISODES_DIR = ROOT_DIR / "episodes"

for d in [CHAR_DIR, HUD_DIR, BGM_DIR, BG_DIR, SFX_DIR, STOCK_DIR, EPISODES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def create_global_registry():
    """Khóa chặt toàn bộ thông số Nhân vật, Màu sắc, Giọng đọc, Bối cảnh và Vũ khí cho 100 tập"""
    registry = {
        "series_title": "NGU HANH CHIEN DOI - NGUYEN GIOI GIAP SU",
        "style_suffix": "cinematic anime tokusatsu background art, detailed sci-fi command center and dramatic battlefields, makoto shinkai lighting, 8k",
        "characters": {
            "hoa_do": {
                "display_name": "HỎA ĐỎ (VIÊM LONG)",
                "role": "Thủ khoa Phân đội Xung Kích - Đội trưởng",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "+2%",
                "pitch": "+2Hz",
                "sub_color": [255, 85, 85],
                "primary_rgb": [220, 40, 40],
                "awake_rgb": [255, 110, 20],
                "hair_rgb": [40, 25, 25]
            },
            "thuy_lam": {
                "display_name": "THỦY LAM (HẢI MINH)",
                "role": "Thủ khoa Phân đội Chiến Thuật - Phó đội",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "-2%",
                "pitch": "-4Hz",
                "sub_color": [90, 200, 255],
                "primary_rgb": [30, 115, 230],
                "awake_rgb": [140, 40, 220],
                "hair_rgb": [20, 35, 65]
            },
            "tho_vang": {
                "display_name": "THỔ VÀNG (THẠCH CƯỜNG)",
                "role": "Thủ khoa Phân đội Hộ Vệ - Tiên phong",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "-5%",
                "pitch": "-8Hz",
                "sub_color": [255, 215, 70],
                "primary_rgb": [225, 175, 30],
                "awake_rgb": [160, 110, 20],
                "hair_rgb": [65, 45, 25]
            },
            "phong_luc": {
                "display_name": "PHONG LỤC (PHONG VÂN)",
                "role": "Thủ khoa Phân đội Trinh Sát",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "+6%",
                "pitch": "+5Hz",
                "sub_color": [100, 245, 130],
                "primary_rgb": [35, 185, 85],
                "awake_rgb": [80, 255, 190],
                "hair_rgb": [25, 60, 40]
            },
            "loi_hong": {
                "display_name": "LÔI HỒNG (LÔI NGỌC)",
                "role": "Thủ khoa Phân đội Công Nghệ",
                "voice": "vi-VN-HoaiMyNeural",
                "rate": "+4%",
                "pitch": "+4Hz",
                "sub_color": [255, 135, 220],
                "primary_rgb": [230, 65, 165],
                "awake_rgb": [190, 90, 255],
                "hair_rgb": [75, 30, 65]
            },
            "long_vu": {
                "display_name": "LONG VŨ (LONG PHÁCH)",
                "role": "Chiến binh thứ 6 - Thiên Tôn Tái Sinh",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "+0%",
                "pitch": "-2Hz",
                "sub_color": [255, 235, 140],
                "primary_rgb": [235, 195, 65],
                "awake_rgb": [255, 250, 190],
                "hair_rgb": [220, 220, 235]
            },
            "chi_huy": {
                "display_name": "CHỈ HUY BẠCH DƯƠNG",
                "role": "Tổng Chỉ Huy A.E.G.I.S. (Quang Phiến)",
                "voice": "vi-VN-HoaiMyNeural",
                "rate": "-2%",
                "pitch": "-3Hz",
                "sub_color": [220, 245, 255],
                "primary_rgb": [210, 225, 245],
                "awake_rgb": [255, 255, 255],
                "hair_rgb": [180, 195, 215]
            },
            "mac_khieu": {
                "display_name": "ĐẠI TƯỚNG MẠC KHIÊU",
                "role": "Thống lĩnh Quân Đoàn Hắc Thực (Hắc Châu)",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "-8%",
                "pitch": "-14Hz",
                "sub_color": [200, 120, 255],
                "primary_rgb": [75, 25, 110],
                "awake_rgb": [180, 20, 90],
                "hair_rgb": [30, 15, 40]
            },
            "quai_vat_tuan": {
                "display_name": "HẮC HÓA THÚ",
                "role": "Chiến binh Quân Đoàn Hắc Thực",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "-6%",
                "pitch": "-11Hz",
                "sub_color": [180, 190, 200],
                "primary_rgb": [85, 95, 105],
                "awake_rgb": [160, 40, 60],
                "hair_rgb": [40, 45, 50]
            },
            "phu_giap": {
                "display_name": "CHIẾN BINH HẮC HÓA",
                "role": "Quân Đoàn Hắc Thực - Tiền Phong",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "-5%",
                "pitch": "-10Hz",
                "sub_color": [195, 150, 230],
                "primary_rgb": [90, 55, 115],
                "awake_rgb": [175, 45, 85],
                "hair_rgb": [45, 35, 55]
            },
            "narrator": {
                "display_name": "NGƯỜI DẪN CHUYỆN",
                "role": "Thuyết minh",
                "voice": "vi-VN-NamMinhNeural",
                "rate": "+0%",
                "pitch": "-2Hz",
                "sub_color": [255, 255, 255],
                "primary_rgb": [200, 215, 235],
                "awake_rgb": [255, 255, 255],
                "hair_rgb": [60, 60, 60]
            }
        },
        "backgrounds": {
            "bg_yuan_world": {
                "prompt": "alien crystal planet temple with giant glowing energy sphere floating in cosmic sky",
                "seed": 1001
            },
            "bg_aegis_arena": {
                "prompt": "futuristic military academy combat testing arena with spectator stands and holographic shields",
                "seed": 1002
            },
            "bg_aegis_command": {
                "prompt": "high tech defense headquarters command bridge with giant tactical world map screen and consoles",
                "seed": 1003
            },
            "bg_aegis_lab": {
                "prompt": "advanced sci-fi weapon forge laboratory with laser containment chambers and elemental cores",
                "seed": 1004
            },
            "bg_city_ruins": {
                "prompt": "modern metropolis street under attack with smoke, sparks and dramatic sunset sky",
                "seed": 1005
            },
            "bg_city_battle": {
                "prompt": "modern metropolis street under attack with smoke, sparks and dramatic sunset sky",
                "seed": 1005
            },
            "bg_quarry_battle": {
                "prompt": "epic rocky mountain quarry battlefield with dramatic dust clouds and sunlight beams",
                "seed": 1006
            },
            "bg_quarry_wasteland": {
                "prompt": "epic rocky mountain quarry wasteland battlefield with dramatic dust clouds and sunlight beams",
                "seed": 1006
            },
            "bg_dark_fortress": {
                "prompt": "sinister alien biomechanical fortress interior with glowing purple dark core in space",
                "seed": 1007
            },
            "bg_mecha_sky": {
                "prompt": "panoramic city skyline with towering clouds and lightning for giant robot battle",
                "seed": 1008
            }
        },
        "bgm_tracks": {
            "bgm_daily": {"file": "assets/bgm/bgm_daily.wav", "duck_ratio": 6, "vol": 0.22},
            "bgm_tension": {"file": "assets/bgm/bgm_tension.wav", "duck_ratio": 5, "vol": 0.26},
            "bgm_crisis": {"file": "assets/bgm/bgm_crisis.wav", "duck_ratio": 8, "vol": 0.20},
            "bgm_awakening": {"file": "assets/bgm/bgm_awakening.wav", "duck_ratio": 4, "vol": 0.30},
            "bgm_awaken": {"file": "assets/bgm/bgm_awakening.wav", "duck_ratio": 4, "vol": 0.30},
            "bgm_battle_rock": {"file": "assets/bgm/bgm_battle_rock.wav", "duck_ratio": 2, "vol": 0.42},
            "bgm_mecha": {"file": "assets/bgm/bgm_mecha.wav", "duck_ratio": 2.5, "vol": 0.38},
            "bgm_mecha_epic": {"file": "assets/bgm/bgm_mecha.wav", "duck_ratio": 2.5, "vol": 0.38},
            "bgm_outro": {"file": "assets/bgm/bgm_outro.wav", "duck_ratio": 6, "vol": 0.24}
        }
    }

    with open(ROOT_DIR / "global_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return registry


def build_character_sprite_pack(char_id, cfg):
    """Vẽ cố định 4 trạng thái (civilian, armored, awakened, broken_helmet) cho từng nhân vật"""
    c_dir = CHAR_DIR / char_id
    c_dir.mkdir(parents=True, exist_ok=True)
    w, h = 600, 750

    p_rgb = tuple(cfg["primary_rgb"]) + (255,)
    a_rgb = tuple(cfg["awake_rgb"]) + (255,)
    h_rgb = tuple(cfg["hair_rgb"]) + (255,)

    # 1. TRẠNG THÁI CIVILIAN (Đồng phục A.E.G.I.S. - Mở mắt & Nhắm mắt)
    for eye_state in ["open", "blink"]:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.polygon([(115, 750), (195, 415), (405, 415), (485, 750)], fill=(30, 38, 52, 255), outline=p_rgb)
        d.polygon([(205, 415), (300, 560), (395, 415)], fill=p_rgb)
        d.polygon([(350, 475), (370, 455), (390, 475), (370, 505)], fill=(255, 215, 90, 255))
        d.ellipse([165, 105, 435, 395], fill=h_rgb)
        d.ellipse([195, 145, 405, 405], fill=(255, 226, 198, 255), outline=(80, 55, 45, 255), width=4)
        d.chord([180, 115, 420, 255], start=180, end=360, fill=h_rgb)
        if eye_state == "open":
            d.ellipse([230, 240, 275, 276], fill=(255, 255, 255, 255), outline=(25, 25, 25, 255), width=3)
            d.ellipse([325, 240, 370, 276], fill=(255, 255, 255, 255), outline=(25, 25, 25, 255), width=3)
            d.ellipse([242, 246, 264, 270], fill=p_rgb)
            d.ellipse([337, 246, 359, 270], fill=p_rgb)
        else:
            d.line([230, 258, 275, 258], fill=(35, 25, 25, 255), width=5)
            d.line([325, 258, 370, 258], fill=(35, 25, 25, 255), width=5)
        img.save(c_dir / f"civ_{eye_state}.png")

    # 3 Khẩu hình miệng dùng chung cho Civilian và Broken Helmet
    m_closed = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(m_closed).line([278, 345, 322, 345], fill=(145, 55, 55, 255), width=5)
    m_closed.save(c_dir / "mouth_closed.png")

    m_small = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(m_small).ellipse([282, 336, 318, 358], fill=(165, 45, 55, 255), outline=(85, 20, 25, 255), width=3)
    m_small.save(c_dir / "mouth_small.png")

    m_wide = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dw = ImageDraw.Draw(m_wide)
    dw.ellipse([270, 325, 330, 372], fill=(190, 35, 45, 255), outline=(85, 20, 25, 255), width=3)
    dw.chord([276, 327, 324, 343], start=180, end=360, fill=(255, 255, 255, 255))
    m_wide.save(c_dir / "mouth_wide.png")

    # 2. TRẠNG THÁI ARMORED & AWAKENED
    for mode_prefix, suit_color, glow_color in [("armor", p_rgb, p_rgb), ("awake", a_rgb, a_rgb)]:
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        db = ImageDraw.Draw(base)
        db.polygon([(100, 750), (175, 405), (425, 405), (500, 750)], fill=suit_color, outline=(240, 240, 250, 255))
        db.polygon([(115, 440), (195, 390), (225, 470), (105, 495)], fill=(235, 200, 80, 255), outline=(40, 40, 40, 255))
        db.polygon([(485, 440), (405, 390), (375, 470), (495, 495)], fill=(235, 200, 80, 255), outline=(40, 40, 40, 255))
        db.polygon([(300, 445), (345, 495), (300, 545), (255, 495)], fill=(255, 255, 255, 255), outline=(255, 215, 0, 255))
        db.ellipse([185, 125, 415, 405], fill=suit_color, outline=(240, 240, 255, 255), width=5)
        db.polygon([(225, 310), (300, 335), (375, 310), (355, 390), (300, 405), (245, 390)], fill=(195, 205, 220, 255), outline=(60, 70, 80, 255))
        db.polygon([(205, 225), (300, 255), (395, 225), (380, 295), (300, 315), (220, 295)], fill=(15, 20, 30, 255), outline=(255, 215, 90, 255))
        db.polygon([(300, 95), (335, 195), (300, 175), (265, 195)], fill=(255, 220, 90, 255))
        base.save(c_dir / f"{mode_prefix}_base.png")

        visor = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dv = ImageDraw.Draw(visor)
        bright_c = (min(255, glow_color[0] + 70), min(255, glow_color[1] + 70), min(255, glow_color[2] + 70), 235)
        dv.polygon([(210, 230), (300, 258), (390, 230), (376, 290), (300, 310), (224, 290)], fill=bright_c)
        dv.polygon([(300, 452), (338, 495), (300, 538), (262, 495)], fill=bright_c)
        visor.save(c_dir / f"{mode_prefix}_visor_glow.png")

        aura = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        da = ImageDraw.Draw(aura)
        aura_alpha = 115 if mode_prefix == "armor" else 175
        ac = (glow_color[0], glow_color[1], glow_color[2], aura_alpha)
        da.ellipse([55, 50, 545, 735], outline=ac, width=16 if mode_prefix == "armor" else 28)
        da.ellipse([85, 80, 515, 710], outline=(255, 255, 255, aura_alpha), width=6)
        aura.save(c_dir / f"{mode_prefix}_aura.png")

    # 3. TRẠNG THÁI BROKEN HELMET
    for eye_state in ["open", "blink"]:
        brk = Image.open(c_dir / "awake_base.png").copy()
        civ_face = Image.open(c_dir / f"civ_{eye_state}.png")
        face_patch = civ_face.crop((215, 215, 390, 385))
        brk.paste(face_patch, (215, 215))
        d_brk = ImageDraw.Draw(brk)
        d_brk.line([(215, 215), (295, 235), (390, 215), (390, 385), (215, 385), (215, 215)], fill=(30, 30, 30, 255), width=5)
        brk.save(c_dir / f"broken_{eye_state}.png")

    # 4. DẢI CẮT ÁNH MẮT (eye_strip.png) cho hiệu ứng Eye Cut-in
    ew, eh = 960, 220
    eye_strip = Image.new("RGBA", (ew, eh), (12, 16, 28, 235))
    de = ImageDraw.Draw(eye_strip)
    for lx in range(0, ew, 48):
        de.line([(lx, 0), (lx - 80, eh)], fill=(p_rgb[0], p_rgb[1], p_rgb[2], 95), width=4)
    de.polygon([(220, 110), (380, 55), (445, 95), (365, 145)], fill=(255, 255, 255, 255), outline=a_rgb, width=4)
    de.polygon([(740, 110), (580, 55), (515, 95), (595, 145)], fill=(255, 255, 255, 255), outline=a_rgb, width=4)
    de.ellipse([335, 75, 385, 125], fill=a_rgb)
    de.ellipse([575, 75, 625, 125], fill=a_rgb)
    eye_strip.save(c_dir / "eye_strip.png")


def create_fixed_hud_overlays():
    """Tạo bộ khung giao diện cố định của Sở Chỉ Huy A.E.G.I.S. và Buồng lái Robot cho cả 100 tập"""
    for mode, (w, h) in [("16x9", (1920, 1080)), ("9x16", (1080, 1920))]:
        hud_aegis = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(hud_aegis)
        cyan = (0, 225, 255, 180)
        d.rectangle([25, 25, w - 25, h - 25], outline=cyan, width=3)
        d.line([(25, 85), (420, 85), (455, 25)], fill=cyan, width=4)
        d.text((45, 42), "A.E.G.I.S. TACTICAL COMMAND NETWORK // ONLINE", fill=(180, 245, 255, 220))
        hud_aegis.save(HUD_DIR / f"hud_aegis_{mode}.png")

        hud_cockpit = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dc = ImageDraw.Draw(hud_cockpit)
        gold = (255, 195, 40, 200)
        dc.polygon([(0, 0), (260, 0), (140, 140), (0, 140)], fill=(20, 25, 35, 190), outline=gold)
        dc.polygon([(w, 0), (w - 260, 0), (w - 140, 140), (w, 140)], fill=(20, 25, 35, 190), outline=gold)
        dc.rectangle([40, 40, w - 40, h - 40], outline=gold, width=4)
        dc.text((w // 2 - 180, 52), "NGU HANH VUONG // CORE SYNCHRONIZED 100%", fill=(255, 230, 120, 230))
        hud_cockpit.save(HUD_DIR / f"hud_cockpit_{mode}.png")


def synthesize_sentai_bgm(filename, chord_freqs, pulse_hz, duration_sec=24):
    """Tạo 7 bản nhạc nền cố định theo đúng nhịp độ 7 Hồi của Super Sentai"""
    sr = 44100
    t = np.linspace(0, duration_sec, sr * duration_sec, endpoint=False)
    signal = np.zeros_like(t)
    pulse = 0.65 + 0.35 * np.sin(2 * np.pi * pulse_hz * t)
    for idx, f in enumerate(chord_freqs):
        harmonic = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * (f * 2) * t)
        signal += (0.25 / (idx + 1)) * pulse * harmonic

    audio_int16 = (signal / np.max(np.abs(signal)) * 13000).astype(np.int16)
    with wave.open(str(BGM_DIR / filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def synthesize_default_sfx():
    """Tổng hợp bộ tiếng động cơ bản (explosion, henshin, sword_slash, power_charge) nếu chưa có file thật"""
    sr = 44100

    def save_wav(name, sig):
        path = SFX_DIR / f"{name}.wav"
        if path.exists():
            return
        peak = np.max(np.abs(sig)) or 1.0
        pcm = (sig / peak * 18000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    rng = np.random.default_rng(2026)
    # 1. explosion.wav (1.4s)
    t_exp = np.linspace(0, 1.4, int(sr * 1.4), endpoint=False)
    noise = rng.normal(0, 1, len(t_exp))
    rumble = np.sin(2 * np.pi * 55 * t_exp) * np.exp(-3.2 * t_exp)
    save_wav("explosion", (noise * np.exp(-4.0 * t_exp) * 0.7 + rumble * 0.8))

    # 2. henshin.wav (1.2s rising synth)
    t_hen = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    freq_h = 220 + 660 * (t_hen / 1.2) ** 1.5
    sig_h = np.sin(2 * np.pi * np.cumsum(freq_h) / sr) * np.sin(np.pi * t_hen / 1.2)
    save_wav("henshin", sig_h)

    # 3. sword_slash.wav (0.45s metallic whoosh)
    t_sw = np.linspace(0, 0.45, int(sr * 0.45), endpoint=False)
    freq_s = 1200 * np.exp(-5.0 * t_sw) + 180
    sig_s = (np.sin(2 * np.pi * np.cumsum(freq_s) / sr) + 0.4 * rng.normal(0, 1, len(t_sw))) * np.exp(-6.5 * t_sw)
    save_wav("sword_slash", sig_s)

    # 4. power_charge.wav (0.9s energy surge)
    t_pc = np.linspace(0, 0.9, int(sr * 0.9), endpoint=False)
    freq_p = 180 + 520 * (t_pc / 0.9)
    sig_p = np.sin(2 * np.pi * np.cumsum(freq_p) / sr) * (0.4 + 0.6 * (t_pc / 0.9))
    save_wav("power_charge", sig_p)


def sync_episode_001_json():
    """Đồng bộ ep_001.json từ thư mục gốc vào episodes/ep_001.json (không ghi đè nếu đã có)"""
    target = EPISODES_DIR / "ep_001.json"
    root_ep = ROOT_DIR / "ep_001.json"
    if root_ep.exists() and not target.exists():
        shutil.copy2(root_ep, target)


if __name__ == "__main__":
    print("1. Đang tạo global_registry.json cố định toàn bộ thông số 100 tập...")
    reg = create_global_registry()

    print("2. Đang vẽ cố định 4 trạng thái Sprite cho toàn bộ nhân vật...")
    for cid, ccfg in reg["characters"].items():
        build_character_sprite_pack(cid, ccfg)

    print("3. Đang tạo giao diện HUD cố định (Sở Chỉ Huy A.E.G.I.S. & Buồng lái Robot)...")
    create_fixed_hud_overlays()

    print("4. Đang tổng hợp 7 bản nhạc nền chuẩn Super Sentai...")
    synthesize_sentai_bgm("bgm_daily.wav", [261.63, 329.63, 392.00], pulse_hz=0.5)
    synthesize_sentai_bgm("bgm_tension.wav", [146.83, 174.61, 220.00], pulse_hz=1.8)
    synthesize_sentai_bgm("bgm_crisis.wav", [110.00, 130.81, 164.81], pulse_hz=0.8)
    synthesize_sentai_bgm("bgm_awakening.wav", [196.00, 246.94, 293.66, 392.00], pulse_hz=2.2)
    synthesize_sentai_bgm("bgm_battle_rock.wav", [164.81, 246.94, 329.63, 493.88], pulse_hz=4.0)
    synthesize_sentai_bgm("bgm_mecha.wav", [130.81, 196.00, 261.63, 392.00], pulse_hz=3.0)
    synthesize_sentai_bgm("bgm_outro.wav", [293.66, 369.99, 440.00], pulse_hz=0.6)

    print("5. Đang tạo bộ tiếng động SFX mặc định (explosion, henshin, sword_slash, power_charge)...")
    synthesize_default_sfx()

    print("6. Đang đồng bộ kịch bản Tập 1 vào episodes/ep_001.json...")
    sync_episode_001_json()
    print("-> HOÀN TẤT KHỞI TẠO VŨ TRỤ NGŨ HÀNH CHIẾN ĐỘI!")
