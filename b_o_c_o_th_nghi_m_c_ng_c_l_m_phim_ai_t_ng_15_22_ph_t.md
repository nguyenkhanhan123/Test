# Báo Cáo Đánh Giá & Thử Nghiệm Hệ Thống Làm Phim AI Tự Động (15–22 Phút)

> **Mục tiêu:** Xây dựng quy trình sản xuất phim AI dài 15–22 phút/tập với chi phí bằng 0 (tận dụng hạ tầng **Google Colab Pro 100 CU/tháng**, **Kaggle Free 30 giờ GPU/tuần**, và **máy tính cá nhân**).  
> **Chiến lược cốt lõi:** Audio-First (làm tiếng trước, dựng hình sau) · Quy tắc 30/70 (30% cảnh động Video AI + 70% cảnh tĩnh lia máy) · Train LoRA 1 lần, dùng cho toàn bộ series.

---

## 1. Tổng Quan Kiến Trúc Pipeline (6 Bước)

Thay vì dựng một trang web cồng kềnh dễ gặp lỗi timeout và tràn RAM, hệ thống hoạt động theo dạng **Pipeline lắp ghép các script/notebook tuần tự**, giao tiếp với nhau qua một thư mục dùng chung trên **Google Drive** và một file kịch bản chuẩn hóa `script.json`.

| Giai đoạn | Nơi chạy tối ưu | Tiêu thụ Colab Pro | Tần suất chạy |
| :--- | :--- | :--- | :--- |
| **1. Kịch bản & Chia Shot (JSON)** | Máy cá nhân / API Free | `0 CU` | Mỗi tập |
| **2. Train LoRA nhân vật & phong cách** | Colab Pro (GPU L4 24GB) | `~2–4 CU / nhân vật` | **1 lần duy nhất** |
| **3. Tạo giọng đọc (TTS) & Đo thời lượng** | Máy cá nhân / Colab T4 | `0–1 CU / tập` | Mỗi tập |
| **4. Sinh ảnh Keyframe (150–250 ảnh)** | Kaggle Free / Colab L4 | `0–3 CU / tập` | Mỗi tập |
| **5. Sinh Video AI (40–60 cảnh động)** | Colab Pro (L4 / A100) | `~12–25 CU / tập` | Mỗi tập |
| **6. Tạo phụ đề & Ghép bản full 20 phút** | Máy cá nhân (CPU) | `0 CU` | Mỗi tập |

---

## 2. Xếp Hạng Các Công Cụ & Thư Viện Tốt Nhất Để Thử Nghiệm

Các công cụ dưới đây được xếp hạng **từ chất lượng cao nhất đến phương án dự phòng nhẹ máy hơn**.

### Bước 1: Kịch bản & Tách cú máy (Shot List)
1. **LLM API miễn phí + `pydantic` (Python):** Dùng mô hình ngôn ngữ tạo kịch bản và dùng `pydantic` ép đầu ra đúng tuyệt đối cấu trúc JSON (không bao giờ lỗi cú pháp khi đưa vào code tự động).
2. **Qwen2.5-14B / 7B Instruct (`ollama` hoặc `transformers`):** Chạy cục bộ không phụ thuộc internet; đặc biệt phù hợp để mở rộng prompt chi tiết cho mô hình video Wan 2.2.

### Bước 2: Train LoRA (Khóa mặt nhân vật & phong cách)
1. **`kohya-ss/musubi-tuner` (Khuyên thử đầu tiên):** Công cụ train LoRA toàn năng nhất hiện nay cho cả ảnh và video (hỗ trợ Wan 2.2, FLUX, HunyuanVideo, LTX). Hỗ trợ `block swap` (đẩy lớp mạng sang RAM CPU) giúp train mượt trên GPU L4 24GB. *(Lưu ý: dùng trọng số nền `bf16`/`fp16` khi train, tránh dùng bản nén `fp8_scaled`).*
2. **`ostris/ai-toolkit`:** Cực kỳ gọn nhẹ để train LoRA nhân vật ảnh tĩnh (đặc biệt dòng FLUX) bằng một file cấu hình `.yaml`, tối ưu VRAM sẵn bằng `qfloat8`.
3. **`Lightricks/LTX-2` (`ltx-trainer`):** Dùng riêng nếu bạn chọn toàn bộ hệ sinh thái LTX để train cả hình ảnh lẫn chuyển động.

### Bước 3: Giọng đọc AI (Text-to-Speech – Ưu tiên tiếng Việt)
1. **`danhtran2mind/Vi-F5-TTS` (Chất lượng cảm xúc số 1 tiếng Việt):** Hỗ trợ nhân bản giọng nói (*voice cloning*) từ đoạn mẫu 5–10 giây. Cho phép gán mỗi nhân vật trong phim một chất giọng riêng biệt, có ngữ điệu tự nhiên.
2. **`Qwen3-TTS` (0.6B / 1.7B, Apache 2.0) hoặc `Chatterbox` (MIT):** Nhân bản giọng đa ngôn ngữ nhanh, độ ổn định cao, giấy phép mở dùng thương mại thoải mái.
3. **`edge-tts` (Nhanh & nhẹ nhất – 0 GPU):** Chạy hoàn toàn trên CPU, tốc độ tạo 20 phút audio chỉ tính bằng chục giây, phát âm tiếng Việt chuẩn và **trả về sẵn mốc thời gian từng từ cho phụ đề**. Nhược điểm: ít cảm xúc bằng F5-TTS, hợp nhất cho giọng người kể chuyện (narrator).
4. **`Kokoro-82M`:** Lựa chọn số 1 chạy trên CPU nếu sản xuất phiên bản **tiếng Anh**.

### Bước 4: Sinh ảnh khung hình chuẩn (Keyframe Generation)
*Trình điều phối khuyên dùng:* **ComfyUI** (chạy dạng Headless/API gọi từ script Python) hoặc thư viện **`diffusers`** của Hugging Face.
1. **`FLUX.2` (`dev` 32B / `klein` 4B):** Đứng đầu về khả năng bám sát prompt phức tạp, ánh sáng điện ảnh và giữ đồng nhất nhân vật khi kết hợp nhiều ảnh tham chiếu. Bản `klein` 4B chạy cực nhanh trên GPU tầm trung (~13GB VRAM).
2. **`Qwen-Image-2512` + `Qwen-Image-Lightning` (Apache 2.0):** Chi tiết chân thực, viết chữ trong ảnh rất chuẩn; bản `Lightning` rút ngắn xuống 4–8 bước sinh ảnh, cực kỳ tiết kiệm GPU khi xuất 200 ảnh.
3. **`SDXL / Illustrious / Pony`:** Lựa chọn tối ưu nhất nếu làm phong cách **Anime / Hoạt hình 2D**, tốc độ cực nhanh trên T4/P100 của Kaggle Free và kho LoRA phong cách phong phú.

### Bước 5: Sinh Video AI (Image-to-Video) & Bộ Thư Viện Tăng Tốc
**Mô hình tạo video:**
1. **`Wan 2.2` (`Wan2.2-I2V-A14B` trên A100 / `Wan2.2-TI2V-5B` trên L4):** Chất lượng chuyển động mượt và thẩm mỹ điện ảnh hàng đầu mã nguồn mở (720p, 24fps). Có thêm nhánh **`Wan2.2-S2V-14B`** chuyên nhép miệng theo file giọng đọc và **`Wan2.2-Animate`** bắt chuyển động nhân vật.
2. **`LTX-2 / LTX-2.3 / LTX-2.5` (`Lightricks/LTX-2`):** Tốc độ render nhanh và có khả năng độc đáo là **tạo đồng thời hình ảnh lẫn âm thanh môi trường (SFX) khớp chuyển động** trong một lượt chạy.

**Bộ 3 thư viện tăng tốc BẮT BUỘC cài kèm (giảm 3–5 lần thời gian đốt CU):**
1. **`sageattention` (SageAttention):** Tăng tốc lớp tính toán Attention bằng INT8 (nhanh hơn 1,5–2,5 lần).
2. **`TeaCache` (ComfyUI Node):** Tái sử dụng kết quả giữa các bước khử nhiễu ít biến đổi (ngưỡng `0.4`), giảm mạnh số phép tính thừa.
3. **`LightX2V` + `Wan 2.2 Distill LoRA (4-step)` + trọng số `fp8`:** Giảm từ ~50 bước suy luận xuống còn **4–8 bước** mỗi clip mà vẫn giữ chất lượng tốt.

### Bước 6: Phụ đề, Hiệu ứng Lia Máy & Ghép Phim Full
1. **`FFmpeg` (`ffmpeg-python`):** Trụ cột xử lý video – chạy CPU máy cá nhân, thêm hiệu ứng Ken Burns (zoom/pan), trộn nhạc nền giảm âm tự động (*audio ducking*), gắn phụ đề và nối clip siêu nhanh không tràn RAM.
2. **`WhisperX` / `faster-whisper`:** Tự động nhận diện và căn mốc thời gian chính xác từng từ cho phụ đề.
3. **`pysubs2`:** Thư viện Python chuyên xử lý, dịch chuyển mốc thời gian (*shift timestamp*) và tạo hiệu ứng màu sắc cho file phụ đề `.ass` / `.srt`.
4. **`MoviePy` (v2):** Chỉ dùng bổ trợ cho các đoạn intro/outro hoặc hiệu ứng chữ phức tạp trên đoạn ngắn (2–3 phút), sau đó dùng FFmpeg ghép tổng.

---

## 3. Giải Mã Cơ Chế Phụ Đề: Phụ Đề Là Gì & Làm Sao Biết Tiếng Đến Đâu Để Thêm?

Đây là phần nhiều người mới làm tự động hóa hay thấy mơ hồ, nhưng thực chất cơ chế hoạt động của nó rất rõ ràng và chuẩn xác nhờ quy trình **Audio-First**.

### 3.1. Phụ đề thực chất là gì?
Phụ đề **không phải hình ảnh**, mà chỉ là một **file văn bản thuần (text)** quy định 3 thông số: câu chữ nào xuất hiện, bắt đầu từ mili-giây thứ mấy, và biến mất ở mili-giây thứ mấy. Có 2 định dạng phổ biến:

* **`.srt` (SubRip Subtitle):** Định dạng cơ bản nhất, chỉ chứa chữ và thời gian.
  ```text
  1
  00:00:00,450 --> 00:00:03,120
  Bầu trời đêm nay đột nhiên đỏ rực một cách kỳ lạ.

  2
  00:00:03,400 --> 00:00:06,050
  Có thứ gì đó vừa rơi xuống khu rừng phía Bắc!
  ```
* **`.ass` (Advanced SubStation Alpha — Khuyên dùng cho phim AI):** Chứa đầy đủ thông tin thiết kế như font chữ, cỡ chữ, viền đen, bóng đổ, vị trí hiển thị và thậm chí **hiệu ứng đổi màu từng chữ kiểu karaoke** khi nhân vật đang đọc.

### 3.2. Làm sao máy tính xác định chính xác "tiếng nói đến đâu" để khớp chữ?

Hệ thống **không đoán mò theo độ dài câu chữ**, mà dựa vào 2 kỹ thuật đọc dữ liệu âm thanh trực tiếp (tùy vào công cụ TTS bạn chọn ở Bước 3):

#### Cách A: Lấy mốc thời gian ngay trong lúc tạo giọng (Dùng với `edge-tts`)
Khi `edge-tts` tổng hợp âm thanh, máy chủ TTS đồng thời gửi về luồng sự kiện **`WordBoundary`** chứa chính xác thời điểm bắt đầu (*offset*) và độ dài (*duration*) của từng từ vừa phát âm.
* **Ưu điểm:** Chính xác 100%, tốc độ tức thì, tốn **0% GPU**, không cần thêm bước phân tích lại âm thanh.
* **Nhược điểm:** Chỉ áp dụng khi dùng `edge-tts` (hoặc các API có hỗ trợ trả về timestamp).

#### Cách B: Căn chỉnh ép buộc từ sóng âm — *Forced Alignment* (Dùng với `Vi-F5-TTS`, `Qwen3-TTS`)
Các mô hình nhân bản giọng mã nguồn mở như `Vi-F5-TTS` chỉ xuất ra file âm thanh thuần (`shot_001.wav`) mà không kèm mốc thời gian. Lúc này, ta đưa file `.wav` đó qua **`WhisperX`** (hoặc `faster-whisper` bật `word_timestamps=True`):
1. **Phát hiện khoảng lặng (VAD – Voice Activity Detection):** Hệ thống quét biểu đồ sóng âm để biết chính xác giây nào có tiếng người nói, giây nào nhân vật ngừng lấy hơi.
2. **Đối chiếu âm vị (Forced Alignment bằng mô hình `wav2vec2`):** Thay vì để AI tự đoán chữ (dễ sai chính tả tiếng Việt), bạn đưa cho nó **file âm thanh `.wav` + đúng câu thoại gốc trong `script.json`**. AI sẽ so khớp từng âm tiết của câu chữ vào đúng đỉnh sóng âm tương ứng, trả về thời điểm bắt đầu và kết thúc của từng từ chính xác tới mức `0.02` giây.

Ví dụ kết quả JSON mà `WhisperX` trả về cho `shot_001.wav`:
```json
[
  {"word": "Bầu", "start": 0.45, "end": 0.68},
  {"word": "trời", "start": 0.69, "end": 0.92},
  {"word": "đêm", "start": 0.94, "end": 1.15},
  {"word": "nay", "start": 1.16, "end": 1.48}
]
```
Từ danh sách từ này, script Python chỉ cần gom cứ 6–10 từ thành 1 dòng phụ đề ngắn để người xem dễ đọc, sau đó lưu ra file `.ass` bằng thư viện `pysubs2`.

### 3.3. Làm sao ghép phụ đề của 200 shot rời vào đúng vị trí trong phim 20 phút?

Vì bạn render từng cú máy riêng biệt (`shot_001.wav`, `shot_002.wav`, ...), mọi file âm thanh lẻ đều bắt đầu từ mốc `0.00s`. Để khớp toàn bộ tập phim dài 20 phút, script Python áp dụng quy tắc **Cộng dồn mốc thời gian toàn cục (Global Offset)**:

$$t_{\text{global}} = t_{\text{local}} + \text{offset}_{\text{tích lũy}}$$

Quy trình chạy tự động bằng Python:
1. Khởi tạo `current_offset = 0.0` (giây).
2. Duyệt lần lượt từng shot trong `script.json`:
   * Dùng `ffprobe` đo chính xác độ dài thực tế của `shot_i.mp4` (hoặc `shot_i.wav` + khoảng nghỉ `0.3s` giữa các câu), gọi là $d_i$.
   * Lấy phụ đề cục bộ của `shot_i`, dùng `pysubs2` dịch chuyển toàn bộ mốc thời gian lên thêm một lượng bằng `current_offset`.
   * Ghi nối các dòng phụ đề đã dịch chuyển vào file tổng `episode_01.ass`.
   * Cập nhật lại mốc tích lũy: $\text{current\_offset} \leftarrow \text{current\_offset} + d_i$.
3. Cuối cùng, dùng **FFmpeg** gắn file `episode_01.ass` vào video đã ghép:
   * **Hardsub (đốt cứng chữ lên hình – luôn hiển thị trên YouTube/TikTok/Facebook):**
     ```bash
     ffmpeg -i episode_01_raw.mp4 -vf "ass=episode_01.ass" -c:a copy episode_01_final.mp4
     ```
   * **Softsub (nhúng luồng phụ đề riêng – người xem tự bật/tắt):**
     ```bash
     ffmpeg -i episode_01_raw.mp4 -i episode_01.srt -c copy -c:s mov_text episode_01_softsub.mp4
     ```

---

## 4. Cấu Trúc `script.json` Mẫu Kết Nối Toàn Bộ Pipeline

Mọi công cụ từ Bước 1 đến Bước 6 đều đọc chung từ một file JSON như sau để tránh sửa tay rời rạc:

```json
{
  "episode_id": "ep_01_pilot",
  "style_lora": "anime_cinematic_v1.safetensors",
  "shots": [
    {
      "shot_id": "shot_001",
      "character": "nhan_vat_a",
      "character_lora": "nhan_vat_a_v1.safetensors",
      "voice_ref": "voices/nhan_vat_a_sample.wav",
      "shot_type": "video_ai",
      "dialogue": "Bầu trời đêm nay đột nhiên đỏ rực một cách kỳ lạ.",
      "image_prompt": "medium shot of nhan_vat_a looking up at a crimson night sky, glowing red clouds, cinematic lighting",
      "motion_prompt": "camera slowly tilts up, wind blowing hair and coat, dramatic clouds moving",
      "duration_sec": null
    },
    {
      "shot_id": "shot_002",
      "character": "narrator",
      "character_lora": null,
      "voice_ref": "edge_tts:vi-VN-NamMinhNeural",
      "shot_type": "ken_burns_zoom_in",
      "dialogue": "Không ai trong thị trấn biết rằng đó là điềm báo của một kỷ nguyên mới.",
      "image_prompt": "wide aerial view of a quiet misty mountain town at midnight under a red sky",
      "motion_prompt": null,
      "duration_sec": null
    }
  ]
}
```
*(Trường `duration_sec` để trống `null` ở Bước 1; sau khi Bước 3 sinh xong file âm thanh, script sẽ tự điền chính xác số giây vào trường này để Bước 5 và Bước 6 render đúng thời lượng).*

---

## 5. Kế Hoạch Thử Nghiệm Thực Chiến (MVP 30 Giây – 5 Shot)

**Tuyệt đối không bắt tay làm ngay tập 20 phút.** Hãy chạy 4 bài kiểm tra nhỏ sau đây (tổng chi phí chỉ tốn khoảng **5–8 CU** trên Colab Pro) để đo tốc độ thực tế và kiểm chứng chất lượng bộ công cụ:

| Giai đoạn Test | Bài kiểm tra cụ thể | Công cụ áp dụng | Tiêu chí ĐẠT (Pass) |
| :--- | :--- | :--- | :--- |
| **Test 1: Âm thanh & Phụ đề** *(Chạy máy cá nhân – 0 CU)* | Tạo 5 file thoại (tổng ~30 giây), tự căn thời gian và xuất 1 file `.ass` tổng. | `Vi-F5-TTS` (hoặc `edge-tts`) + `WhisperX` + `pysubs2` | Giọng đọc rõ, không méo tiếng; file `.ass` khớp từng câu với sai số dưới `0.15s`. |
| **Test 2: Khóa mặt nhân vật** *(Colab L4 – ~2–3 CU)* | Chuẩn bị 20 ảnh 1 nhân vật chính, train 1 LoRA, sinh thử 5 ảnh Keyframe theo 5 góc máy khác nhau. | `musubi-tuner` (hoặc `ai-toolkit`) + `FLUX.2` / `SDXL` | Cả 5 ảnh giữ đúng khuôn mặt và trang phục nhân vật chính trên 85%. |
| **Test 3: Sinh Video AI tốc độ cao** *(Colab L4/A100 – ~2–4 CU)* | Lấy 2 ảnh từ Test 2 chạy Image-to-Video (mỗi clip 5 giây), bật đủ bộ tăng tốc. | `Wan 2.2` (`I2V-A14B` hoặc `TI2V-5B`) + `SageAttention` + `TeaCache` + `fp8` | Thời gian render dưới 3 phút/clip; nhân vật không bị biến dạng mặt khi cử động. |
| **Test 4: Ghép phim tự động** *(Chạy máy cá nhân – 0 CU)* | Ghép 2 clip AI + 3 ảnh tĩnh hiệu ứng Ken Burns + 5 file thoại + nhạc nền + đốt phụ đề `.ass`. | `FFmpeg` (`ffmpeg-python`) | Xuất ra file `pilot_30s.mp4` hoàn chỉnh, âm thanh không bị lệch hình, chữ dễ đọc. |

---

## 6. Nguyên Tắc An Toàn Khi Chạy Trên Google Colab Pro

1. **Tự động ngắt GPU ngay khi chạy xong:** Luôn dán đoạn code này vào ô cuối cùng của mọi Notebook để tránh quên tắt máy làm đốt sạch 100 CU qua đêm:
   ```python
   from google.colab import runtime
   runtime.unassign()
   ```
2. **Lưu checkpoint & đầu ra trực tiếp về Google Drive:** Mount Google Drive ngay từ ô đầu tiên và trỏ thư mục `output` sang Drive. Khi render 50 cảnh động, nếu phiên Colab bị ngắt giữa chừng ở cảnh 32, script chỉ cần kiểm tra file đã tồn tại trên Drive để chạy tiếp từ cảnh 33.
3. **Chốt nguyên tắc san tải:** Toàn bộ khâu tạo ảnh Keyframe (Bước 4) nên đẩy sang **Kaggle Notebooks (30 giờ GPU miễn phí/tuần)** để dành trọn 100 CU của Colab Pro cho việc train LoRA (Bước 2) và render Video AI độ phân giải 720p (Bước 5).