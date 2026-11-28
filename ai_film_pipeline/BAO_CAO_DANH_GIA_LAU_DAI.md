# Báo Cáo Nghiên Cứu & Đánh Giá Lâu Dài — Thử Nghiệm MVP `pilot_30s.mp4`

> **Sản phẩm thử nghiệm:** `output/pilot_30s.mp4` (`36.50 giây`, `1280x720`, `24 fps`, `876 frames`, `8.75 MiB` / `9,177,801 bytes`, H.264 + AAC stereo 48kHz, hardsub `.ass` karaoke tiếng Việt).  
> **Mục tiêu đánh giá:** Kiểm chứng toàn bộ quy trình **Audio-First + Quy tắc 30/70** trước khi đầu tư thời gian & Compute Units (CU) làm tập dài 15–22 phút.

---

## 1. Kết Quả Thực Nghiệm Từ Bản Thử Nghiệm MVP

| Hạng mục kiểm tra | Công cụ thực tế đã chạy | Kết quả đo được | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Bước 1: Kịch bản & Schema** | Python + `pydantic` (`step1_script.py`, `schema.py`) | `6 shot` (2 `video_ai` = 33% / 4 `ken_burns_*` = 67%), validate chặt chẽ | **ĐẠT** — Schema nối xuyên suốt không sửa tay |
| **Bước 3: Giọng đọc & Mốc từ** | `edge-tts 7.3.1` (`boundary="WordBoundary"`) | 3 giọng phân vai (`NamMinhNeural` narrator `-3Hz/-4%`, `minh_khang` `+4Hz/+3%`, `HoaiMyNeural` `linh_chi` `+1Hz/+2%`), khớp 100/100 từ | **ĐẠT** — Tốc độ rất nhanh, 0 GPU |
| **Bước 4: Đồng nhất Keyframe** | Tạo Reference Sheet (`cast_sheet.png`) trước, rồi sinh 6 Keyframe 16:9 theo ref | Giữ nhất quán trang phục, kiểu tóc, kẹp tóc đỏ, móc cổ bạc, la bàn đồng xuyên suốt 6 góc máy | **ĐẠT** — Khẳng định vai trò khóa reference trước |
| **Bước 5: Cảnh động AI (`30%`)** | *Máy cục bộ không có NVIDIA GPU (`torch CPU`)* → dùng bộ mô phỏng 2.5D (`step6_render_local.py`) | Render được chuyển động camera, sương mù, ánh sáng đỏ & tàn lửa bay (`shot_001`, `shot_005`) | **MÔ PHỎNG CỤC BỘ** — Cần chạy `cloud_scripts/step5_batch_wan22_i2v.py` trên Colab L4/A100 để đánh giá biểu cảm/nhân vật chuyển động thật từ `Wan 2.2` |
| **Bước 6: Ghép phim, BGM/SFX & Phụ đề** | `imageio-ffmpeg` (`libass`, `zoompan`, `sidechaincompress`) + `pysubs2` + `scipy` | `876 frames` chính xác tuyệt đối `36.500s`, 0 frame lệch, 15 câu phụ đề karaoke không chồng chéo, nhạc nền tự nhỏ khi thoại | **ĐẠT** — Sẵn sàng scale lên tập 20 phút |

---

## 2. Các Lỗi Tiềm Ẩn Đã Phát Hiện & Tự Động Khắc Phục Trong Code

Đây là những điểm **nếu không thử nghiệm thực tế sẽ gây hỏng hàng loạt khi làm phim 20 phút (~200 shot)**:

1. **Lỗi trôi khung hình (Cumulative Drift) ở mốc 20 phút:**
   - Độ dài file `.mp3`/`.wav` hiếm khi khớp bội số nguyên của `1/24s`. Sai số `~0.02s/shot` nhân cho `200 shot` sẽ gây **lệch tiếng và phụ đề tới 4 giây** ở cuối tập.
   - **Giải pháp đã code sẵn:** `step3_tts.py` làm tròn lên theo lưới `24fps` (`quantize_to_fps`), còn `step6_subtitles.py` đo trực tiếp số frame thực tế (`count_video_frames`) của từng file `.mp4` trước khi cộng dồn `Global Offset`.
2. **Thay đổi mặc định của `edge-tts` bản 7.x:**
   - Từ `edge-tts 7.x`, tham số mặc định là `boundary="SentenceBoundary"`, **không trả về từng từ** như các hướng dẫn cũ. Phải truyền rõ `boundary="WordBoundary"`.
   - Ngoài ra `WordBoundary` của Edge TTS **cắt bỏ dấu câu** (`"đâu"`, `"xem"` thay vì `"đâu."`, `"xem!"`) → đã viết hàm `_restore_punctuation()` trong `step6_subtitles.py` để khôi phục nguyên vẹn dấu câu từ `dialogue` và tách cụm theo khoảng nghỉ tự nhiên (`>= 0.35s`) thay vì để chữ rơi rớt 1 từ cuối dòng.
3. **Chồng chữ phụ đề khi không nối khoảng nghỉ:**
   - Kéo dài `end + 0.14s` để chữ đỡ tắt đột ngột khiến dòng trước chồng lên dòng sau (đã phát hiện ngay khi soi frame `shot_001_mid.png` lần 1). Đã khóa cứng `prev.end <= next.start - 0.02s` và thêm kiểm thử tự động trong `tests/test_pipeline.py`.
4. **Rung hình (Jitter) của FFmpeg `zoompan`:**
   - Chạy `zoompan` trực tiếp trên ảnh `1280x720` sẽ bị giật từng pixel theo bước nguyên. Đã xử lý bằng cách **siêu lấy mẫu 3x (`3840x2160` Lanczos)** trước khi `zoompan` về `1280x720` — chuyển động lia máy mượt như After Effects.
5. **Thiếu `ffprobe` trên Windows mặc định:**
   - Gói `imageio-ffmpeg` chỉ kèm `ffmpeg.exe`, không có `ffprobe.exe`. Toàn bộ code đã dùng `media_utils.py` đọc trực tiếp từ `ffmpeg -i` và `mutagen` nên chạy ổn định trên mọi máy Windows không cần cài thêm binary ngoài.

---

## 3. Kết Quả Nghiên Cứu Đối Chiếu Stack Cloud GPU (2026)

Khi mở rộng lên **tập dài 15–22 phút** bằng Colab Pro & Kaggle Free, cần lưu ý 5 phát hiện kỹ thuật quan trọng từ nghiên cứu:

1. **Chi phí Compute Units (CU) thực tế trên Google Colab Pro (100 CU/tháng):**
   - **L4 (24GB VRAM):** `~1.8 – 2.4 CU/giờ` — tỷ lệ hiệu năng/giá tốt nhất cho pipeline.
   - **A100 (40/80GB VRAM):** `~10.5 – 15.0 CU/giờ` — đắt gấp **5–6 lần L4**, chỉ nên dùng nếu cần `Wan 2.2 A14B` không offload hoặc `LTX-2` full.
   - *Dự toán 1 tập 20 phút (45 cảnh `video_ai` @ 5s/cảnh trên L4 với LightX2V 4–6 bước + SageAttention + TeaCache):* khoảng `2.5 – 4.0 giờ` → **`~6 – 10 CU/tập`**, hoàn toàn đủ làm **8–12 tập/tháng** trong gói 100 CU.
2. **Kiến trúc 2 nhánh của `Wan 2.2 A14B` khi train LoRA (`musubi-tuner`):**
   - `Wan 2.2` tách làm 2 mô hình **High-Noise** và **Low-Noise**. Để khóa mặt và trang phục nhân vật, cần pre-cache latents + `umt5-xxl` trước và train LoRA nhánh **Low-Noise** (hoặc dùng `Wan 2.2 TI2V-5B` nhẹ hơn trên L4).
3. **Bẫy VRAM & Precision của `LTX-2` và Kaggle P100/T4:**
   - `LTX-2 / 2.3` (22B) yêu cầu gốc **32GB+ VRAM**, nên chạy trên L4 24GB **bắt buộc dùng FP8 + CPU offload + pipeline distilled 8/4 bước**.
   - Kaggle Free GPU (`T4`/`P100` 16GB) không có tensor core FP8 native; script `cloud_scripts/step4_batch_keyframes_flux2.py` đã cài cơ chế tự chọn `float16` trên T4/P100 và dùng `FLUX.2 [klein] 4B` (hoặc `Qwen-Image-2512 Lightning` GGUF / `SDXL`).
4. **`WhisperX` tiếng Việt thiếu alignment model mặc định:**
   - Nếu nâng cấp sang `Vi-F5-TTS` hoặc `Gwen-TTS`, chạy `whisperx --language vi` mặc định sẽ báo lỗi thiếu model `wav2vec2`. Phải chỉ định rõ: `--align_model nguyenvulebinh/wav2vec2-large-vi-vlsp2020`.
5. **Bản quyền thương mại TTS tiếng Việt:**
   - Nhiều trọng số cộng đồng của `Vi-F5-TTS` kế thừa giấy phép **phi thương mại (`CC-BY-NC-SA-4.0`)**. Nếu định kiếm tiền lâu dài trên YouTube/TikTok, nên ưu tiên **`Gwen-TTS` (fine-tune từ `Qwen3-TTS-0.6B` Apache 2.0)** hoặc `edge-tts` cho giọng dẫn chuyện, và chỉ dùng `Vi-F5-TTS` khi kiểm tra kỹ license bộ checkpoint cụ thể.

---

## 4. Kết Luận Khả Năng Đánh Giá Lâu Dài

- **Tính khả thi kỹ thuật:** **Rất cao (9/10).** Toàn bộ khung xương tự động hóa (`script.json` → TTS + WordBoundary → Keyframe → FFmpeg Ken Burns / 2.5D / Video AI → Audio Ducking → Hardsub `.ass`) đã chạy thông suốt, không cần thao tác dựng tay trên CapCut/Premiere.
- **Chất lượng hình ảnh & đồng bộ:** Các Keyframe giữ nhất quán nhân vật xuất sắc khi có `cast_sheet.png`; phụ đề tiếng Việt hiển thị sắc nét, đúng nhịp từng cụm từ.
- **Bước tiếp theo để chốt 100% trước khi sản xuất tập 1 (15–22 phút):**
  1. Xem trực tiếp `output/pilot_30s.mp4` để đánh giá nhịp cắt cảnh, giọng đọc `edge-tts` và độ đọc dễ của phụ đề.
  2. Chạy thử duy nhất **2 file ảnh `shot_001.png` và `shot_005.png`** qua `cloud_scripts/step5_batch_wan22_i2v.py` trên Colab L4 (`~1–2 CU`), chép 2 clip `.mp4` kết quả vào `assets/video_ai/`, rồi chạy lại `python run_all.py --skip-script --skip-tts` để xem bản ghép thật hoàn chỉnh giữa **Video AI `Wan 2.2` (30%) + Ken Burns (70%)**.
