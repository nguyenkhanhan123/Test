HƯỚNG DẪN THU ÂM GIỌNG MẪU BIỂU CẢM (CHO VI-F5-TTS / GWEN-TTS)

Để giọng AI clone nghe giống bạn VÀ biểu đạt đúng cảm xúc từng hoàn cảnh, hãy thu 4–5 đoạn ngắn (mỗi đoạn 6–10 giây, phòng yên tĩnh, nói rõ chữ) đặt vào thư mục này (`user_inputs/voices/`):

1. `calm.wav` + `calm.txt`
   -> Giọng kể chuyện / nói chuyện bình tĩnh.
2. `tense.wav` + `tense.txt`
   -> Giọng căng thẳng, cảnh báo, hơi gấp hoặc hạ giọng.
3. `angry_action.wav` + `angry_action.txt`
   -> Giọng hô to, gằn mạnh, dứt khoát dùng cho cảnh chiến đấu / hành động.
4. `whisper.wav` + `whisper.txt`
   -> Giọng nói nhỏ / thì thầm bí ẩn.
5. `sad.wav` + `sad.txt`
   -> Giọng trầm buồn, chậm rãi.

LƯU Ý CỰC QUAN TRỌNG:
- Mỗi file `.txt` phải ghi **chính xác từng chữ** bạn đã đọc trong file `.wav` tương ứng (Vi-F5-TTS cần `ref_text` khớp 100% âm thanh mẫu để giữ nguyên dấu thanh tiếng Việt).
