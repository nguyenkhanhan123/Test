# Hướng Dẫn Chạy Cloud GPU (Colab Pro & Kaggle Free)

Tài liệu tóm tắt kỹ thuật đã kiểm chứng cho quy trình dài hạn (15–22 phút/tập):

## 1. Ngân sách Compute Units (CU) thực tế trên Google Colab Pro (100 CU/tháng)
- **T4 (16GB VRAM):** ~`1.6 – 2.0 CU/giờ` (quá chậm cho Wan 2.2 14B, chỉ nên dùng TTS/F5-TTS).
- **L4 (24GB VRAM — Khuyến nghị chính):** ~`1.8 – 2.4 CU/giờ` (tối ưu nhất giữa chi phí CU và tốc độ; Ada Lovelace hỗ trợ FP8/BF16 native).
- **A100 (40GB/80GB VRAM):** ~`10.5 – 15.0 CU/giờ` (đốt CU nhanh gấp 5–6 lần L4; chỉ bật nếu render cảnh phức tạp cần VRAM lớn).

## 2. Lưu ý quan trọng khi triển khai dài hạn (đã cập nhật từ nghiên cứu thực tế)
1. **Wan 2.2 A14B có 2 model (High-Noise + Low-Noise):**
   Khi train LoRA bằng `musubi-tuner`, cần pre-cache latents + `umt5-xxl` trước, và train nhánh **Low-Noise** trước để khóa khuôn mặt/trang phục nhân vật.
2. **Kaggle Free GPU (T4x2 hoặc P100 16GB):**
   P100 và T4 **không hỗ trợ native FP8 tensor core**, P100 cũng không hỗ trợ native `bfloat16` hiệu năng cao — script `step4_batch_keyframes_flux2.py` tự động chuyển về `float16` khi chạy trên T4/P100. Với `FLUX.2 [klein] 4B` hoặc `SDXL/Illustrious`, Kaggle Free 30 giờ/tuần đủ sức sinh 200+ ảnh Keyframe/tập mà tốn `0 CU`.
3. **WhisperX tiếng Việt:**
   WhisperX bản chuẩn **không có sẵn model forced-alignment mặc định cho `--language vi`**. Khi chuyển sang `Vi-F5-TTS` hoặc `Gwen-TTS`, bắt buộc truyền thêm:
   `--align_model nguyenvulebinh/wav2vec2-large-vi-vlsp2020`
4. **Giấy phép thương mại cho TTS tiếng Việt:**
   - `Vi-F5-TTS` bản cộng đồng thường mang license phi thương mại (`CC-BY-NC-SA-4.0`) do bộ dữ liệu huấn luyện.
   - Nếu sản xuất thương mại lâu dài cần nhân bản giọng tiếng Việt, ưu tiên **`Gwen-TTS` (từ Qwen3-TTS 0.6B Apache 2.0)** hoặc giọng narrator bằng `edge-tts`.
5. **TeaCache khi ghép mặt nhân vật:**
   Ngưỡng `0.40` cho tốc độ nhanh ở cảnh rộng/hành động, nhưng với cảnh cận mặt nên hạ xuống `0.25` để tránh mất nét khuôn mặt giữa các frame.
