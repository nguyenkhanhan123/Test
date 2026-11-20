# KIẾN TRÚC NÂNG CẤP SẢN XUẤT SUPER SENTAI HIỆN ĐẠI (100% MIỄN PHÍ – CHUẨN TOKUSATSU)
## DỰ ÁN: NGŨ HÀNH CHIẾN ĐỘI – NGUYÊN GIỚI GIÁP SƯ

Tài liệu này cung cấp giải pháp nâng cấp toàn diện từ mô hình "truyện tranh tĩnh đọc thoại" sang **Hệ thống Sản xuất Tokusatsu Hiện Đại bằng Code (0 Đồng)**. Mục tiêu cốt lõi là đáp ứng sự khắt khe của dòng phim siêu nhân: **hình ảnh sắc nét đồng bộ 100 tập, góc máy hành động biến hóa linh hoạt, hiệu ứng va chạm vật lý có lực và giọng lồng tiếng bùng nổ cảm xúc** mà không tốn chi phí bản quyền hay thuê phần mềm.

---

## PHẦN 1: HỆ SINH THÁI CÔNG CỤ MIỄN PHÍ 100% THAY THẾ ĐỒ THÔ

Để bộ phim có "hồn" mà không tốn tiền, nguyên tắc vàng là **không dùng code để vẽ nhân vật**, mà dùng **AI mã nguồn mở miễn phí tạo nguyên liệu gốc chất lượng cao**, sau đó dùng **Python + FFmpeg làm Đạo diễn dựng phim tự động**.

| Khâu Sản Xuất | Công Cụ Miễn Phí Khuyên Dùng | Vai Trò Trong Dự Án 100 Tập |
| :--- | :--- | :--- |
| **1. Tạo hình Nhân vật & Giáp** | **Pollinations AI (Flux)** hoặc **Google AI Studio** + thư viện **`rembg`** | Tạo ảnh nhân vật 2D Anime/Tokusatsu sắc nét (khóa cố định `seed`) và tự động xóa phông nền trong suốt bằng 1 dòng lệnh Python. |
| **2. Diễn xuất Đời thường** | **LivePortrait** hoặc **EchoMimic** *(Chạy free trên Google Colab T4)* | Biến ảnh chân dung đời thường thành video nhân vật biết nghiêng đầu, nhíu mày, chớp mắt và nhép môi khớp từng âm tiết thoại. |
| **3. Cảnh Biến thân & Robot** | **Wan 2.1 (1.3B / 14B)** hoặc **LTX-Video** *(Chạy free trên Google Colab T4)* | Tạo sẵn **12–15 clip hành động tái sử dụng (Stock Footage)** dài 4–5 giây (Biến thân, Điểm danh nổ khói, Hợp thể Robot) dùng chung cho cả 100 tập. |
| **4. Giọng Lồng Tiếng** | **RVC v2** + **F5-TTS (Tiếng Việt)** + **Edge-TTS** + **FFmpeg Audio Filter** | Tách biệt hoàn toàn: Giọng đời thường giàu cảm xúc, giọng vang qua bộ đàm mũ bảo hiểm khi mặc giáp và tiếng gầm bạo tẩu/chiêu thức. |
| **5. Nhạc Phim (OST) & Tiếng Động (SFX)** | **Suno AI (Gói Free hàng ngày)** + **Pixabay Sound Effects** | Tạo nhạc nền J-Rock chiến đấu, nhạc giao hưởng Robot và tải kho tiếng nổ, chém kiếm, còi báo động miễn phí bản quyền. |

---

## PHẦN 2: GIẢI QUYẾT BÀI TOÁN KHẮT KHE NHẤT – GIỌNG LỒNG TIẾNG SIÊU NHÂN (0 ĐỒNG)

Phim siêu nhân sẽ mất 70% khí thế nếu dùng một giọng đọc AI đều đều từ đầu đến cuối. Bạn hãy áp dụng **Quy trình Lồng Tiếng 3 Tầng Miễn Phí** dưới đây:

### Tầng 1: Kỹ thuật "Tự Thu Âm + RVC v2" Cho Các Cảnh Cao Trào (Hét Biến Thân, Bạo Tẩu)
* **Nguyên lý:** AI đọc văn bản rất khó tự tạo ra tiếng thở dốc, tiếng gầm đau đớn khi Hỏa Đỏ mất kiểm soát *Nham Thể* (Tập 44) hay tiếng hô dứt khoát *"Hỏa Giáp – Biến Thân!"*.
* **Cách làm 0 đồng:**
  1. Bạn dùng chính điện thoại của mình tự thu âm giọng diễn (chỉ cần đúng nhịp điệu, hơi thở và độ gằn giọng, không cần giọng hay).
  2. Cài **RVC v2 (Retrieval-based Voice Conversion – mã nguồn mở miễn phí)** hoặc chạy trên Google Colab. Đưa file thu âm của bạn qua mô hình giọng Anime/Phim hành động (1 giọng Nam trầm ấm cho Hỏa Đỏ, 1 giọng Nam lạnh lùng cho Thủy Lam, 1 giọng Nữ sắc sảo cho Lôi Hồng, 1 giọng Trầm khàn cho Mạc Khiêu).
  3. RVC giữ nguyên 100% cảm xúc và tiếng gầm của bạn nhưng thay đổi hoàn toàn màu giọng thành nhân vật chuyên nghiệp.

### Tầng 2: Nhân Bản Cảm Xúc Tự Động Với F5-TTS / VieNeu-TTS (Cho Các Tập Dài)
* Với các đoạn đối thoại dài giữa tập, bạn dùng **F5-TTS (bản fine-tune tiếng Việt trên HuggingFace)** hoặc **VieNeu-TTS**.
* Chuẩn bị sẵn **3 file âm thanh mẫu dài 5 giây** cho mỗi nhân vật:
  * `char_calm.wav`: Giọng bình tĩnh phân tích chiến thuật.
  * `char_angry.wav`: Giọng căng thẳng, phẫn nộ khi gặp nguy hiểm.
  * `char_shout.wav`: Giọng hô lớn quyết tâm trên chiến trường.
* Khi kịch bản gắn nhãn cảm xúc nào, Engine tự động nạp đúng file mẫu đó để AI đọc khớp tông cảm xúc.

### Tầng 3: Xử Lý Vật Lý Âm Thanh Bằng FFmpeg (Tạo Chất Giọng Mũ Bảo Hiểm)
Trong phim Super Sentai, ngay khoảnh khắc nhân vật biến hình xong, giọng nói của họ **bắt buộc phải đổi sang chất giọng vang qua bộ đàm bên trong mũ bảo hiểm**. Engine ở Phần 5 tích hợp sẵn 3 bộ lọc FFmpeg tự động:
1. **`civilian` (Ngoài đời):** Giọng gốc tự nhiên, ấm áp.
2. **`helmet_radio` (Mặc Nguyên Giáp / Nguyên Thể):** Lọc cắt dải trầm dưới $300\text{ Hz}$ và dải cao trên $3.500\text{ Hz}$, nén động học và thêm độ vang kim loại ngắn $12\text{ ms}$ (`highpass=f=300,lowpass=f=3500,acompressor,aecho=0.8:0.88:12:0.25`).
3. **`villain_boss` (Mạc Khiêu / Quái vật Hắc Hóa):** Hạ cao độ xuống $14\%$ và tạo độ vang không gian lớn (`asetrate=44100*0.86,aresample=44100,aecho=0.8:0.85:35:0.35`).

---

## PHẦN 3: BỘ CÂU LỆNH (PROMPTS) TẠO HÌNH & STOCK FOOTAGE ĐỒNG BỘ 100 TẬP

### 3.1. Tạo Kho Ảnh Nhân Vật Gốc & Dải Cắt Ánh Mắt (`eye_strip.png`)
Bạn chỉ cần làm bước này **1 lần duy nhất** trên **Pollinations.ai** hoặc **Google AI Studio** (miễn phí), sau đó dùng thư viện `rembg` tách nền và lưu vào `assets/characters/<char_id>/`:

* **Prompt mẫu tạo Nguyên Giáp (Ví dụ Hỏa Đỏ – `armor_base.png`):**
  > `full body character design of a red tokusatsu super sentai ranger warrior, sleek metallic crimson and gold armor, glowing black visor with sharp phoenix crest on helmet, heroic battle stance, isolated on solid plain white background, sharp anime cel-shaded art style, 8k --seed 202601`
  *(Thay `red/crimson` và `phoenix` thành: `blue/cobalt/dragon` cho Thủy Lam; `yellow/amber/lion` cho Thổ Vàng; `green/emerald/falcon` cho Phong Lục; `pink/magenta/lightning` cho Lôi Hồng; `royal gold and white/imperial dragon` cho Long Vũ).*

* **Prompt mẫu tạo Trạng thái Đột phá Nguyên Thể (Ví dụ Nham Thể – `awake_base.png`):**
  > `full body tokusatsu super sentai red warrior upgraded form, dark obsidian volcanic armor plates with glowing molten lava cracks, blazing magma aura, heavy shoulder pauldrons, isolated on solid plain white background, dynamic anime tokusatsu art, 8k --seed 202601`

* **Prompt mẫu tạo Dải Cắt Ngang Ánh Mắt (`eye_strip.png` – dùng khi Điểm danh & Tung chiêu):**
  > `extreme close up horizontal narrow strip of determined anime hero eyes glowing with fiery crimson energy, dramatic speed lines, intense tokusatsu anime cut-in frame, 8k --seed 202601`

### 3.2. Tạo 10 Clip Hành Động Tái Sử Dụng (`Stock Footage`) Trên Google Colab Free
Đặc sản giúp Super Sentai tiết kiệm chi phí mà vẫn cực kỳ hoành tráng là **tái sử dụng các cảnh quay chuẩn mực**. Hãy mở notebook **Wan 2.1** hoặc **LTX-Video** trên Google Colab (chọn GPU T4 miễn phí) để tạo sẵn 10 clip `.mp4` (mỗi clip 4–5 giây) lưu vào `assets/stock_clips/`:
1. `henshin_fire.mp4` / `water` / `earth` / `wind` / `thunder`: Cảnh kích hoạt thiết bị biến thân, giáp cơ học bay tới lắp ráp lên người.
2. `rollcall_explosion.mp4`: Cảnh 5 chiến binh đứng tạo dáng điểm danh và phía sau bùng nổ 5 cột khói màu Đỏ – Xanh – Vàng – Lục – Hồng.
3. `gokai_slash_finisher.mp4`: Cảnh thanh kiếm tụ năng lượng Ngũ Hành chém ra luồng sóng năng lượng rực sáng.
4. `mecha_combine.mp4`: Cảnh 5 cỗ máy Nguyên Thú lắp ráp thành Robot khổng lồ **Ngũ Hành Vương**.
5. `mecha_sword_finish.mp4`: Cảnh Robot Ngũ Hành Vương vung đại kiếm chém đôi quái vật khổng lồ kèm vụ nổ rực trời.

---

## PHẦN 4: HỆ THỐNG 6 GÓC MÁY ĐIỆN ẢNH (`shot_type`) THAY THẾ KHUNG HÌNH TĨNH

Thay vì ép mọi câu thoại vào một bố cục đứng Trái/Phải nhàm chán, Engine mới cho phép kịch bản JSON điều phối **6 loại góc máy hành động** liên tục:

1. **`establishing` (Toàn cảnh thiết lập):** Camera lướt chậm qua bối cảnh rộng (Hành tinh Nguyên Giới vỡ nát, Toàn cảnh Học viện A.E.G.I.S., Thành phố bốc cháy); không đặt nhân vật đè lên giữa màn hình, dùng cho lời dẫn chuyện hoặc lời thoại qua bộ đàm.
2. **`portrait_drama` (Cận cảnh diễn xuất đời thường):** Phóng to nhân vật (hoặc chèn trực tiếp video biểu cảm từ **LivePortrait**), có khung tên nhân vật hiện đại ở góc dưới bên trái (*Lower-Third*).
3. **`com_link` (Chia khung bộ đàm chiến thuật):** Màn hình tự động cắt làm 2 hoặc 3 khung dọc song song khi các chiến binh liên lạc khẩn cấp giữa trận chiến hoặc hội ý tại Sở Chỉ Huy.
4. **`action_clash` (Giao chiến vật lý tốc độ cao):** Nhân vật tấn công lao xé gió từ trái sang phải (có hiệu ứng nhòe chuyển động *Motion Blur*), nhân vật phòng thủ giật lùi khi trúng đòn, kèm **Impact Frame** (chớp khung hình trắng – đảo màu âm bản – đen trắng tương phản cao trong $0.12\text{ giây}$) và rung chấn camera (`shake`).
5. **`eye_cutin` (Dải cắt ngang ánh mắt):** Trước khi tung đòn tất sát hoặc đột phá lên *Nham, Thực, Trọng, Bão, Chấn*, một dải băng phóng to đôi mắt rực sáng của chiến binh trượt ngang màn hình kèm tiếng rít năng lượng.
6. **`stock_clip` (Chèn cảnh quay hành động `.mp4`):** Tự động chèn clip biến thân, nổ khói điểm danh hoặc hợp thể Robot từ thư mục `assets/stock_clips/`, tự động khớp độ dài với giọng hô chiêu thức và đóng dấu phụ đề lên trên.

---

## PHẦN 5: MÃ NGUỒN ENGINE NÂNG CẤP HOÀN CHỈNH (`modern_sentai_engine.py`)

Hãy lưu đoạn mã dưới đây thành file **`modern_sentai_engine.py`** trong thư mục dự án của bạn. Engine này chạy trực tiếp trên CPU hoặc GPU, tự động hóa toàn bộ 6 góc quay, bộ lọc giọng mũ bảo hiểm, hiệu ứng *Impact Frame*, *Eye Cut-in* và trộn tiếng động *SFX*.

### Cài đặt thư viện cần thiết:
```bash
pip install edge-tts Pillow numpy pysubs2 mutagen requests
# Tùy chọn: cài thêm rembg để tự động xóa phông nền ảnh AI tải về
pip install rembg onnxruntime
```

### Mã nguồn `modern_sentai_engine.py`:
```python
import os
import json
import wave
import argparse
import asyncio
import subprocess
import urllib.parse
from pathlib import Path
import requests
import numpy as np
import edge_tts
import pysubs2
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter, ImageFont

FPS = 24
ROOT_DIR = Path(".")
ASSETS_DIR = ROOT_DIR / "assets"
CHAR_DIR = ASSETS_DIR / "characters"
BG_DIR = ASSETS_DIR / "backgrounds"
STOCK_DIR = ASSETS_DIR / "stock_clips"
SFX_DIR = ASSETS_DIR / "sfx"
HUD_DIR = ASSETS_DIR / "ui_hud"
BUILD_DIR = ROOT_DIR / "build_modern"

for d in [CHAR_DIR, BG_DIR, STOCK_DIR, SFX_DIR, HUD_DIR, BUILD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_dims(mode: str) -> tuple[int, int]:
    return (1080, 1920) if mode == "tiktok" else (1920, 1080)


# ==============================================================================
# 1. MODULE XỬ LÝ GIỌNG LỒNG TIẾNG SIÊU NHÂN (MŨ BẢO HIỂM & PHẢN DIỆN)
# ==============================================================================
def process_tokusatsu_voice(raw_mp3: Path, final_wav: Path, voice_filter: str = "none"):
    """
    Chuyển đổi MP3 sang WAV 48kHz và áp dụng bộ lọc âm thanh đặc trưng của Super Sentai:
    - 'none': Giọng đời thường tự nhiên.
    - 'helmet': Giọng vang qua bộ đàm trong mũ bảo hiểm chiến đấu.
    - 'villain': Giọng trầm khàn vang vọng uy nghiêm cho Mạc Khiêu / Quái vật.
    - 'radio': Giọng loa phát thanh tại Sở Chỉ Huy A.E.G.I.S.
    """
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
        ["ffmpeg", "-y", "-i", str(raw_mp3), "-af", af, "-ac", "1", "-ar", "48000", str(final_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )


def get_rms_envelopes(wav_path: Path, fps: int = FPS) -> np.ndarray:
    """Đo biên độ âm lượng từng khung hình để điều khiển khẩu hình / hào quang giáp"""
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


async def generate_voice_and_subtitles(
    text: str,
    char_cfg: dict,
    voice_filter: str,
    raw_mp3: Path,
    proc_wav: Path,
    ass_path: Path,
    mode: str,
    sem: asyncio.Semaphore
) -> float:
    """Sinh giọng đọc (hoặc nhận file thu âm RVC có sẵn) và xuất phụ đề .ass đổi màu theo nhân vật"""
    async with sem:
        # Nếu bạn đã tự thu âm qua RVC v2 và đặt sẵn file raw_mp3, code sẽ giữ nguyên file cảm xúc đó!
        if not raw_mp3.exists() or raw_mp3.stat().st_size < 500:
            comm = edge_tts.Communicate(
                text=text,
                voice=char_cfg.get("voice", "vi-VN-NamMinhNeural"),
                rate=char_cfg.get("rate", "+0%"),
                pitch=char_cfg.get("pitch", "+0Hz")
            )
            with open(raw_mp3, "wb") as f:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

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
            marginv=240 if mode == "tiktok" else 60,
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
# 2. MODULE HIỆU ỨNG TOKUSATSU HIỆN ĐẠI (IMPACT FRAME, EYE CUT-IN, LOWER-THIRD)
# ==============================================================================
def apply_impact_frame(canvas: Image.Image, f_idx: int) -> Image.Image:
    """Chớp 3 khung hình va chạm trắng -> đảo màu âm bản -> đen trắng tương phản cao"""
    if f_idx == 0:
        return Image.new("RGBA", canvas.size, (255, 255, 255, 240))
    elif f_idx == 1:
        return ImageOps.invert(canvas.convert("RGB")).convert("RGBA")
    elif f_idx == 2:
        gray = ImageOps.grayscale(canvas).point(lambda p: 255 if p > 125 else 0)
        return gray.convert("RGBA")
    return canvas


def draw_eye_cutin_banner(canvas: Image.Image, char_id: str, theme_rgb: tuple, progress: float) -> Image.Image:
    """Dải băng phóng to ánh mắt chiến binh trượt ngang màn hình khi Điểm danh / Tung chiêu"""
    if progress > 0.65:
        return canvas
    w, h = canvas.size
    band_h = int(h * 0.22)
    y_mid = int(h * 0.40)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = theme_rgb

    slide_x = int((1.0 - min(1.0, progress * 3.2)) * w)
    draw.rectangle([slide_x, y_mid - band_h // 2, w + slide_x, y_mid + band_h // 2], fill=(r, g, b, 215))
    draw.line([(0, y_mid - band_h // 2), (w, y_mid - band_h // 2)], fill=(255, 255, 255, 255), width=6)
    draw.line([(0, y_mid + band_h // 2), (w, y_mid + band_h // 2)], fill=(255, 255, 255, 255), width=6)

    eye_file = CHAR_DIR / char_id / "eye_strip.png"
    if eye_file.exists():
        eye_im = Image.open(eye_file).convert("RGBA").resize((int(w * 0.65), band_h - 12), Image.Resampling.LANCZOS)
        overlay.alpha_composite(eye_im, dest=(int(w * 0.18) + slide_x, y_mid - band_h // 2 + 6))

    return Image.alpha_composite(canvas, overlay)


def draw_lower_third_tag(canvas: Image.Image, display_name: str, role_text: str, theme_rgb: tuple):
    """Vẽ bảng tên nhân vật hiện đại ở góc dưới bên trái trong các cảnh cận diễn xuất"""
    if not display_name or display_name == "NGƯỜI DẪN CHUYỆN":
        return
    w, h = canvas.size
    box_w, box_h = 520, 86
    x0, y0 = 50, h - 220

    tag_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tag_layer)
    r, g, b = theme_rgb
    d.polygon([(x0, y0), (x0 + box_w, y0), (x0 + box_w - 30, y0 + box_h), (x0, y0 + box_h)], fill=(15, 20, 32, 210))
    d.rectangle([x0, y0, x0 + 12, y0 + box_h], fill=(r, g, b, 255))

    try:
        f_main = ImageFont.truetype("arialbd.ttf", 28)
        f_sub = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        f_main = ImageFont.load_default()
        f_sub = ImageFont.load_default()

    d.text((x0 + 26, y0 + 12), display_name, fill=(255, 255, 255, 255), font=f_main)
    d.text((x0 + 26, y0 + 48), role_text, fill=(min(255, r + 90), min(255, g + 90), min(255, b + 90), 240), font=f_sub)
    canvas.alpha_composite(tag_layer)


# ==============================================================================
# 3. MODULE QUẢN LÝ TÀI NGUYÊN & RENDER 6 GÓC MÁY LINH HOẠT
# ==============================================================================
def ensure_background(bg_id: str, bg_info: dict, style_suffix: str, mode: str) -> Path:
    w, h = get_dims(mode)
    out_jpg = BG_DIR / f"{bg_id}_{mode}.jpg"
    if out_jpg.exists() and out_jpg.stat().st_size > 5000:
        return out_jpg

    prompt = f"{bg_info['prompt']}, {style_suffix}"
    seed = bg_info.get("seed", 2026)
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={w}&height={h}&seed={seed}&nologo=true&model=flux"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 5000:
            with open(out_jpg, "wb") as f:
                f.write(resp.content)
            return out_jpg
    except Exception:
        pass

    # Nền Gradient điện ảnh dự phòng khi mất mạng
    img = Image.new("RGB", (w, h), (18, 24, 42))
    draw = ImageDraw.Draw(img)
    for y in range(0, h, 4):
        draw.rectangle([0, y, w, y + 4], fill=(int(18 + 45 * y / h), int(24 + 55 * y / h), int(48 + 80 * y / h)))
    img.save(out_jpg, quality=92)
    return out_jpg


def load_character_art(char_id: str, state: str, target_w: int, target_h: int, flip: bool = False) -> Image.Image:
    """
    Nạp ảnh nhân vật chất lượng cao từ assets/characters/<char_id>/<state>.png.
    Nếu bạn chưa bỏ ảnh AI vào, tự động sinh khung đại diện có nhãn rõ ràng để test kịch bản ngay.
    """
    c_dir = CHAR_DIR / char_id
    c_dir.mkdir(parents=True, exist_ok=True)
    img_path = c_dir / f"{state}.png"

    if img_path.exists():
        im = Image.open(img_path).convert("RGBA").resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        im = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([40, 40, target_w - 40, target_h - 20], radius=30, fill=(35, 45, 70, 220), outline=(255, 210, 80, 255), width=4)
        d.text((target_w // 2, target_h // 2), f"{char_id.upper()}\n[{state}]", fill=(255, 255, 255, 255), anchor="mm")

    return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else im


def render_modern_scene(
    scene: dict,
    chars_reg: dict,
    bg_jpg: Path,
    proc_wav: Path,
    ass_path: Path,
    out_mp4: Path,
    mode: str
):
    """Render phân cảnh linh hoạt theo 6 shot_type: establishing, portrait_drama, com_link, action_clash, eye_cutin, stock_clip"""
    if out_mp4.exists() and out_mp4.stat().st_size > 10000:
        return

    w, h = get_dims(mode)
    shot_type = scene.get("shot_type", "portrait_drama")
    envelopes = get_rms_envelopes(proc_wav, fps=FPS)
    total_frames = len(envelopes) + int(FPS * 0.35)
    total_dur = round(total_frames / FPS, 2)
    escaped_ass = ass_path.resolve().as_posix().replace(":", r"\:")

    # TRƯỜNG HỢP ĐẶC BIỆT: CHÈN CLIP HÀNH ĐỘNG STOCK FOOTAGE (.MP4) CÓ SẴN
    stock_file = STOCK_DIR / scene.get("clip_name", "")
    if shot_type == "stock_clip" and stock_file.exists():
        raw_scene = out_mp4.with_suffix(".raw.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(stock_file),
            "-i", str(proc_wav),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},ass='{escaped_ass}',format=yuv420p",
            "-af", "apad=pad_dur=0.35",
            "-t", str(total_dur),
            "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(raw_scene)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        mix_sfx_and_finalize(raw_scene, scene.get("sfx"), out_mp4)
        return

    # CÁC GÓC MÁY RENDER ĐỘNG BẰNG ENGINE
    scale_w, scale_h = int(w * 1.15), int(h * 1.15)
    bg_raw = Image.open(bg_jpg).convert("RGBA").resize((scale_w, scale_h), Image.Resampling.LANCZOS)
    max_dx, max_dy = scale_w - w, scale_h - h

    speaker_id = scene.get("speaker", "hoa_do")
    spk_cfg = chars_reg.get(speaker_id, {"display_name": "", "role": "", "primary_rgb": [255, 200, 80]})
    theme_rgb = tuple(spk_cfg.get("primary_rgb", [255, 200, 80]))
    state = scene.get("state", "civilian")
    camera = scene.get("camera", "zoom_in")

    # Chuẩn bị nhân vật theo từng loại góc quay
    main_char_img = None
    target_char_img = None
    split_imgs = []

    if shot_type in ("portrait_drama", "eye_cutin"):
        cw, ch = int(w * 0.42), int(h * 0.88)
        main_char_img = load_character_art(speaker_id, state, cw, ch)
    elif shot_type == "action_clash":
        cw, ch = int(w * 0.36), int(h * 0.78)
        attacker_id = scene.get("attacker", speaker_id)
        defender_id = scene.get("defender", "quai_vat_tuan")
        main_char_img = load_character_art(attacker_id, scene.get("attacker_state", state), cw, ch, flip=False)
        target_char_img = load_character_art(defender_id, scene.get("defender_state", "armor_base"), cw, ch, flip=True)
    elif shot_type == "com_link":
        panel_ids = scene.get("panel_chars", ["hoa_do", "thuy_lam", "phong_luc"])
        pw = w // len(panel_ids)
        for pid in panel_ids:
            split_imgs.append((pid, load_character_art(pid, state, int(pw * 0.95), int(h * 0.75))))

    hud_img = None
    if scene.get("show_hud", False):
        h_file = HUD_DIR / ("hud_aegis_16x9.png" if mode == "youtube" else "hud_aegis_9x16.png")
        if h_file.exists():
            hud_img = Image.open(h_file).convert("RGBA")

    raw_scene = out_mp4.with_suffix(".raw.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgba", "-r", str(FPS),
        "-i", "-",
        "-i", str(proc_wav),
        "-vf", f"ass='{escaped_ass}',format=yuv420p",
        "-af", "apad=pad_dur=0.35",
        "-t", str(total_dur),
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(raw_scene)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    for idx in range(total_frames):
        prog = idx / max(1, total_frames - 1)
        vol = envelopes[idx] if idx < len(envelopes) else 0.0

        # Chuyển động Camera
        if camera == "shake" or (shot_type == "action_clash" and idx < 10):
            amp = 22 if idx < 10 else 5
            cx = int(np.clip(max_dx // 2 + np.random.uniform(-amp, amp), 0, max_dx))
            cy = int(np.clip(max_dy // 2 + np.random.uniform(-amp, amp), 0, max_dy))
        elif camera == "crash_zoom":
            fp = min(1.0, idx / (FPS * 0.4))
            cx, cy = int(max_dx * fp * 0.5), int(max_dy * fp * 0.5)
        elif camera == "pan_right":
            cx, cy = int(max_dx * prog), max_dy // 2
        else:
            cx, cy = int(max_dx * prog * 0.5), int(max_dy * prog * 0.5)

        canvas = bg_raw.crop((cx, cy, cx + w, cy + h)).copy()

        # Vẽ theo từng góc quay điện ảnh
        if shot_type in ("portrait_drama", "eye_cutin") and main_char_img:
            breath_y = int(np.sin(idx * 0.16) * 4)
            pos_x = (w - main_char_img.width) // 2
            pos_y = h - main_char_img.height + 20 + breath_y
            canvas.alpha_composite(main_char_img, dest=(pos_x, pos_y))
            draw_lower_third_tag(canvas, spk_cfg.get("display_name", ""), spk_cfg.get("role", ""), theme_rgb)
            if shot_type == "eye_cutin":
                canvas = draw_eye_cutin_banner(canvas, speaker_id, theme_rgb, prog)

        elif shot_type == "action_clash" and main_char_img and target_char_img:
            # Nhân vật tấn công lao xé gió từ trái sang phải trong 0.35 giây đầu
            dash_p = min(1.0, idx / (FPS * 0.35))
            atk_x = int(60 + dash_p * (w * 0.25))
            def_x = int(w - target_char_img.width - 80 + (15 if idx >= 6 else 0))
            base_y = h - main_char_img.height + 20
            canvas.alpha_composite(target_char_img, dest=(def_x, base_y))
            canvas.alpha_composite(main_char_img, dest=(atk_x, base_y))
            # Kích hoạt Impact Frame (khung hình 6, 7, 8 ngay lúc chạm trán)
            if scene.get("impact_frame", True) and idx in (6, 7, 8):
                canvas = apply_impact_frame(canvas, idx - 6)

        elif shot_type == "com_link" and split_imgs:
            # Chia khung dọc đa nhân vật kiểu truyện tranh hành động
            panel_w = w // len(split_imgs)
            for p_idx, (pid, p_im) in enumerate(split_imgs):
                px0 = p_idx * panel_w
                sub_bg = canvas.crop((px0, 0, px0 + panel_w, h))
                if pid != speaker_id:
                    sub_bg = ImageEnhance.Brightness(sub_bg.convert("RGB")).enhance(0.55).convert("RGBA")
                sub_bg.alpha_composite(p_im, dest=((panel_w - p_im.width) // 2, h - p_im.height + 30))
                canvas.paste(sub_bg, (px0, 0))
                if p_idx > 0:
                    ImageDraw.Draw(canvas).line([(px0, 0), (px0, h)], fill=(255, 255, 255, 230), width=6)

        if hud_img:
            canvas.alpha_composite(hud_img)

        if scene.get("flash_start", False) and idx < 4:
            fa = int(220 * (1.0 - idx / 4.0))
            canvas.alpha_composite(Image.new("RGBA", (w, h), (255, 255, 255, fa)))

        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    proc.wait()
    mix_sfx_and_finalize(raw_scene, scene.get("sfx"), out_mp4)


def mix_sfx_and_finalize(raw_mp4: Path, sfx_name: str, final_mp4: Path):
    """Ghép tiếng động chiến đấu (explosion.wav, slash.wav, henshin.wav) vào phân cảnh"""
    sfx_path = SFX_DIR / f"{sfx_name}.wav" if sfx_name else None
    if sfx_path and sfx_path.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(raw_mp4),
            "-i", str(sfx_path),
            "-filter_complex", "[1:a]volume=0.70[sfx];[0:a][sfx]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(final_mp4)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        raw_mp4.unlink(missing_ok=True)
    else:
        raw_mp4.replace(final_mp4)
```

---

## PHẦN 6: VÍ DỤ KỊCH BẢN JSON ĐÃ NÂNG CẤP (ĐA GÓC QUAY & LỌC GIỌNG MŨ BẢO HIỂM)

Dưới đây là cách bạn viết kịch bản trong file `ep_xxx.json` để khai thác tối đa **6 góc quay điện ảnh** và **bộ lọc giọng mũ bảo hiểm**:

```json
{
  "act_id": "act_05",
  "chapter_title": "HỒI V: NGUYÊN GIÁP XUẤT KÍCH",
  "bgm_id": "bgm_battle_rock",
  "scenes": [
    {
      "shot_type": "establishing",
      "bg_id": "bg_city_battle",
      "camera": "pan_right",
      "speaker": "narrator",
      "voice_filter": "none",
      "sfx": "explosion",
      "text": "Khói lửa bao trùm đại lộ trung tâm khi Hắc Hóa Thú tung đòn hủy diệt!"
    },
    {
      "shot_type": "com_link",
      "bg_id": "bg_city_battle",
      "panel_chars": ["hoa_do", "thuy_lam", "phong_luc"],
      "state": "civilian",
      "speaker": "hoa_do",
      "voice_filter": "none",
      "text": "Thủy Lam, Phong Lục! Thiết bị biến thân Nguyên Khí đã đồng bộ hoàn tất rồi!"
    },
    {
      "shot_type": "stock_clip",
      "clip_name": "henshin_fire.mp4",
      "speaker": "hoa_do",
      "voice_filter": "none",
      "sfx": "henshin",
      "text": "NGUYÊN KHÍ KÍCH HOẠT — HỎA GIÁP BIẾN THÂN!"
    },
    {
      "shot_type": "eye_cutin",
      "bg_id": "bg_city_battle",
      "speaker": "hoa_do",
      "state": "armor_base",
      "voice_filter": "helmet",
      "camera": "crash_zoom",
      "flash_start": true,
      "sfx": "power_charge",
      "text": "Ngọn lửa thiêu rụi mọi bóng tối! HỎA ĐỎ — HỎA GIÁP!"
    },
    {
      "shot_type": "action_clash",
      "bg_id": "bg_city_battle",
      "attacker": "hoa_do",
      "attacker_state": "armor_base",
      "defender": "quai_vat_tuan",
      "defender_state": "armor_base",
      "speaker": "hoa_do",
      "voice_filter": "helmet",
      "impact_frame": true,
      "sfx": "sword_slash",
      "text": "Nhận lấy chiêu này! HỎA LONG XUNG KÍCH QUYỀN!"
    }
  ]
}
```