"""Colab Pro (L4 24GB) Step 2 Template: Train character/style LoRA with musubi-tuner.

Reflects verified 2026 best practices:
- Pre-cache VAE latents & UMT5-XXL embeddings before training to prevent OOM on 24GB L4.
- Train in bf16/fp16 base precision (avoid compressed fp8_scaled base during LoRA training).
- Note Wan 2.2 A14B two-expert (High-Noise + Low-Noise) split: train Low-Noise first
  for facial & costume identity lock.
- Always auto-unassign Colab GPU runtime on completion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/ai_film_series")
DATASET_TOML = DRIVE_ROOT / "datasets/minh_khang/dataset.toml"
MODELS_DIR = DRIVE_ROOT / "models/wan22"
LORA_OUT_DIR = DRIVE_ROOT / "loras"


def run_cmd(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_wan22_character_lora(
    character_name: str = "minh_khang_v1",
    expert: str = "low_noise",  # "low_noise" (face/costume lock) or "high_noise" (motion)
    max_steps: int = 1200,
    blocks_to_swap: int = 20,
) -> None:
    dit_file = (
        MODELS_DIR / "wan2.2_i2v_low_noise_14B_bf16.safetensors"
        if expert == "low_noise"
        else MODELS_DIR / "wan2.2_i2v_high_noise_14B_bf16.safetensors"
    )
    vae_file = MODELS_DIR / "wan_2.1_vae.safetensors"
    t5_file = MODELS_DIR / "umt5_xxl_fp16.safetensors"

    # 1. Pre-cache latents and text embeddings (saves ~14GB VRAM during training)
    run_cmd([
        "python", "wan_cache_latents.py",
        "--dataset_config", str(DATASET_TOML),
        "--vae", str(vae_file),
    ])
    run_cmd([
        "python", "wan_cache_text_encoder_outputs.py",
        "--dataset_config", str(DATASET_TOML),
        "--t5", str(t5_file),
        "--batch_size", "4",
    ])

    # 2. Train LoRA with block swapping + 8-bit AdamW for L4 24GB
    output_name = f"{character_name}_{expert}"
    run_cmd([
        "accelerate", "launch", "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "wan_train_network.py",
        "--task", "i2v-A14B",
        "--dit", str(dit_file),
        "--dataset_config", str(DATASET_TOML),
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--blocks_to_swap", str(blocks_to_swap),
        "--gradient_checkpointing",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", "1e-4",
        "--network_module", "networks.lora_wan",
        "--network_dim", "32",
        "--network_alpha", "16",
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.0",
        "--max_train_steps", str(max_steps),
        "--save_every_n_steps", "300",
        "--output_dir", str(LORA_OUT_DIR),
        "--output_name", output_name,
    ])


if __name__ == "__main__":
    try:
        train_wan22_character_lora()
    finally:
        # Safety guard: prevent burning remaining Colab CU after completion/error
        try:
            from google.colab import runtime  # type: ignore

            runtime.unassign()
        except ImportError:
            pass
