"""Prepare user-supplied photos & emotional voice clips for Colab LoRA & Action training."""

from __future__ import annotations

import argparse
from pathlib import Path

from schema import EmotionTag, EpisodeScript, Shot, ShotType, VoiceEmotionSample, VoiceSpec

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare_dataset_and_action_script(base_dir: Path, trigger_word: str = "nv_chinh") -> None:
    photos_dir = base_dir / "user_inputs" / "photos"
    voices_dir = base_dir / "user_inputs" / "voices"
    photos_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in photos_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(photos)} character photo(s) in {photos_dir}")
    if len(photos) < 10:
        print("  [TIP] Upload 15-20 varied face/half-body/full-body photos for >90% LoRA consistency.")

    # Auto-create matching caption .txt files for Kohya / musubi-tuner / ai-toolkit
    for p in photos:
        cap = p.with_suffix(".txt")
        if not cap.exists():
            cap.write_text(
                f"{trigger_word}, realistic cinematic portrait, natural lighting, sharp focus",
                encoding="utf-8",
            )

    # Auto-select bf16 on Ada/Ampere (L4/A100) or fp16 on Turing (T4) / CPU
    train_dtype = "fp16"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            train_dtype = "bf16"
    except ImportError:
        pass

    # Generate ostris/ai-toolkit LoRA training config YAML (SDXL default: zero HF token gate, fits T4 & L4 smoothly)
    lora_yaml = base_dir / "user_inputs" / "train_lora_config.yaml"
    lora_yaml.write_text(
        f"""job: extension
config:
  name: "{trigger_word}_lora_v1"
  process:
    - type: sd_trainer
      training_folder: "{(base_dir / 'loras').as_posix()}"
      device: cuda:0
      trigger_word: "{trigger_word}"
      network:
        type: lora
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 300
        max_step_saves_to_keep: 3
      datasets:
        - folder_path: "{photos_dir.as_posix()}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 1200
        gradient_accumulation_steps: 2
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: constant
        dtype: {train_dtype}
      model:
        # Un-gated SDXL base (works on T4/L4 immediately; swap to black-forest-labs/FLUX.1-dev + is_flux: true + quantize: true if HF logged in)
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_v2: false
        base_weights: null
      sample:
        prompt: "{trigger_word}, dynamic martial arts combat stance in rain, dramatic crimson night lighting, anime film still"
        width: 1024
        height: 576
        steps: 25
        samples_every: 300
""",
        encoding="utf-8",
    )

    # Scan emotional voice recordings
    emotion_bank: dict[EmotionTag, VoiceEmotionSample] = {}
    for emo in EmotionTag:
        wav = voices_dir / f"{emo.value}.wav"
        txt = voices_dir / f"{emo.value}.txt"
        if wav.exists() and txt.exists():
            emotion_bank[emo] = VoiceEmotionSample(
                ref_audio=f"user_inputs/voices/{wav.name}",
                ref_text=txt.read_text(encoding="utf-8").strip(),
            )
    print(f"Found {len(emotion_bank)} emotion voice sample(s): {[e.value for e in emotion_bank]}")

    # Build ready-to-customize 5-shot Action Scene script template
    hero_voice = VoiceSpec(
        engine="vi_f5_tts" if emotion_bank else "edge_tts",
        voice_id="user_inputs/voices/calm.wav" if emotion_bank else "vi-VN-NamMinhNeural",
        emotion_bank=emotion_bank,
    )
    action_script = EpisodeScript(
        episode_id="ep_action_pilot",
        title="Thử Nghiệm Cảnh Hành Động & Giọng Biểu Cảm",
        fps=24,
        width=1280,
        height=720,
        characters={"hero": hero_voice},
        shots=[
            Shot(
                shot_id="shot_001",
                character="hero",
                character_lora=f"{trigger_word}_lora_v1.safetensors",
                voice_ref="vi_f5_tts:calm",
                voice_spec=hero_voice,
                shot_type=ShotType.KEN_BURNS_ZOOM_IN,
                emotion=EmotionTag.WHISPER,
                dialogue="Giữ im lặng... chúng đã bao vây toàn bộ khu kho rồi.",
                image_prompt=f"cinematic medium close-up of {trigger_word} peeking around a dark concrete pillar in rain, tense focused eyes, moody blue neon rim lighting",
                sfx_cue="wind_ominous",
            ),
            Shot(
                shot_id="shot_002",
                character="hero",
                character_lora=f"{trigger_word}_lora_v1.safetensors",
                voice_ref="vi_f5_tts:tense",
                voice_spec=hero_voice,
                shot_type=ShotType.VIDEO_AI,
                emotion=EmotionTag.TENSE,
                dialogue="Không còn đường lui nữa. Chuẩn bị đi!",
                image_prompt=f"dynamic low-angle shot of {trigger_word} dropping into a martial arts combat stance on wet asphalt, dramatic sparks and rain",
                motion_prompt="camera rapidly pushes in, rain droplets splashing, character shifts weight into combat stance",
                sfx_cue="wind_ominous",
            ),
            Shot(
                shot_id="shot_003",
                character="hero",
                character_lora=f"{trigger_word}_lora_v1.safetensors",
                voice_ref="vi_f5_tts:angry_action",
                voice_spec=hero_voice,
                shot_type=ShotType.VIDEO_AI,
                emotion=EmotionTag.ANGRY_ACTION,
                dialogue="Tránh ra ngay!",
                image_prompt=f"high-energy action shot of {trigger_word} mid-lunge delivering a powerful spinning strike, motion blur, flying sparks",
                motion_prompt="fast dynamic tracking camera following explosive martial arts strike and dodge, debris flying",
                end_keyframe_path="assets/keyframes/shot_003_end.png",
                driving_video_path="user_inputs/stunt_ref_003.mp4",
                sfx_cue="meteor_impact",
            ),
            Shot(
                shot_id="shot_004",
                character="hero",
                character_lora=f"{trigger_word}_lora_v1.safetensors",
                voice_ref="vi_f5_tts:calm",
                voice_spec=hero_voice,
                shot_type=ShotType.KEN_BURNS_ZOOM_OUT,
                emotion=EmotionTag.CALM,
                dialogue="Tạm thời thoát rồi... nhưng đây mới chỉ là khởi đầu.",
                image_prompt=f"wide atmospheric rear-quarter shot of {trigger_word} standing atop a rain-slicked rooftop catching breath overlooking glowing city skyline",
                sfx_cue="portal_hum",
            ),
        ],
    )
    out_json = base_dir / "script_action.json"
    out_json.write_text(action_script.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote action template: {out_json} and LoRA config: {lora_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", default="nv_chinh", help="LoRA trigger token for the user")
    args = parser.parse_args()
    prepare_dataset_and_action_script(Path(__file__).resolve().parent, trigger_word=args.trigger)
