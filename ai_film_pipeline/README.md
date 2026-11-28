# AI Film Automation Pipeline (Audio-First 30/70 Architecture)

Thư mục này triển khai thực nghiệm toàn bộ kiến trúc 6 bước từ báo cáo `b_o_c_o_th_nghi_m_c_ng_c_l_m_phim_ai_t_ng_15_22_ph_t.md`:

## Cấu trúc thư mục
- `schema.py` — Hợp đồng dữ liệu Pydantic v2 cho `script.json` (liên kết Step 1 → Step 6).
- `step1_script.py` — Sinh và kiểm định kịch bản `script.json` (MVP 6 shot, tỷ lệ 33% `video_ai` / 67% `ken_burns_*`).
- `step3_tts.py` — Tổng hợp giọng đọc tiếng Việt (`edge-tts`, bật `boundary="WordBoundary"`), đo thời lượng thực tế, chuẩn hóa WAV 48kHz và khóa thời lượng từng shot theo lưới `24fps`.
- `step6_subtitles.py` — Khôi phục dấu câu từ `dialogue`, tách cụm tự nhiên theo khoảng nghỉ & dấu câu, cộng dồn `Global Offset` theo đúng số frame thực tế của từng clip video để xuất `.ass` (karaoke `\kf`) và `.srt`.
- `sfx_bgm.py` — Tổng hợp nhạc nền điện ảnh D-minor và hiệu ứng âm thanh (SFX) đồng bộ theo timeline.
- `step6_render_local.py` — Render Ken Burns siêu lấy mẫu 3x chống rung (`zoompan`), mô phỏng chuyển động 2.5D cục bộ khi chưa gắn clip `Wan 2.2` từ Colab, trộn audio tự động giảm nhạc nền (`sidechaincompress`), và xuất `output/pilot_30s.mp4`.
- `cloud_scripts/` — Bộ script chuẩn hóa cho các bước GPU đám mây:
  - `step2_train_lora_musubi.py` (Colab Pro L4 24GB — `musubi-tuner` cho Wan 2.2 High/Low-Noise + block swapping)
  - `step4_batch_keyframes_flux2.py` (Kaggle Free P100/T4 hoặc Colab L4 — sinh ảnh hàng loạt có resume)
  - `step5_batch_wan22_i2v.py` (Colab Pro L4/A100 — `Wan 2.2` + `LightX2V` + `SageAttention` + `TeaCache`)
- `tests/test_pipeline.py` — Bộ kiểm thử tự động kiểm tra schema, độ lệch thời lượng video/audio và toàn vẹn phụ đề `.ass`.

## Cách chạy nhanh cục bộ (Windows CPU)
```powershell
python d:\Tiktok\ai_film_pipeline\step1_script.py
python d:\Tiktok\ai_film_pipeline\step3_tts.py
python d:\Tiktok\ai_film_pipeline\step6_render_local.py
python d:\Tiktok\ai_film_pipeline\tests\test_pipeline.py
```

Nếu đã có clip AI render từ Colab (`Wan 2.2` / `LTX-2`), đặt vào `assets/video_ai/shot_001.mp4`, `assets/video_ai/shot_005.mp4` trước khi chạy `step6_render_local.py` — pipeline sẽ tự động nhận và thay thế bản mô phỏng 2.5D cục bộ.
