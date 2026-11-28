"""Colab T4 (15GB) & L4 (24GB) Step 5 Runnable Script: Wan 2.2 TI2V-5B Action Generator.

Adapts precision, VAE dtype, CPU offload mode, and resolution/frames automatically
based on detected GPU VRAM so both Tesla T4 (15GB) and NVIDIA L4 (24GB) succeed
without Out-Of-Memory crashes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def get_gpu_profile() -> tuple[torch.dtype, int, int, bool]:
    """Return (dtype, max_area, max_frames, is_low_vram_t4)."""
    if not torch.cuda.is_available():
        return torch.float32, 384 * 672, 41, True
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 20.0 or not torch.cuda.is_bf16_supported():
        # Tesla T4 (15GB, Turing): native FP16, 480x768, 49 frames (~2-3s video), sequential offload
        return torch.float16, 448 * 768, 49, True
    # NVIDIA L4 (24GB, Ada): native BF16, 480x832, up to 81-121 frames
    return torch.bfloat16, 480 * 832, 81, False


def render_wan22_action_shots(
    base_dir: Path,
    script_path: Path,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    steps: int = 24,
) -> None:
    script = json.loads(script_path.read_text(encoding="utf-8"))
    out_dir = base_dir / "assets" / "video_ai"
    out_dir.mkdir(parents=True, exist_ok=True)

    ai_shots = [s for s in script["shots"] if s["shot_type"] == "video_ai"]
    pending = [s for s in ai_shots if not (out_dir / f"{s['shot_id']}.mp4").exists()]
    if not pending:
        print("All video_ai shots already rendered.")
        return

    dtype, max_area, max_frames_cap, is_t4 = get_gpu_profile()
    print(f"Loading {model_id} (dtype={dtype}, low_vram_t4={is_t4}, max_frames={max_frames_cap})...")

    # Float32 VAE avoids black/NaN artifacts when running Wan 2.2 in FP16 on T4
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    if is_t4:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    neg_prompt = (
        "static frame, frozen body, slow motion drift, deformed hands, extra limbs, "
        "blurry face, low quality, text, watermark, subtitle"
    )

    for shot in pending:
        shot_id = shot["shot_id"]
        kf_path = base_dir / shot["keyframe_path"]
        out_mp4 = out_dir / f"{shot_id}.mp4"

        image = load_image(str(kf_path))
        aspect = image.height / image.width
        height = int(round(np.sqrt(max_area * aspect)) // mod_value * mod_value)
        width = int(round(np.sqrt(max_area / aspect)) // mod_value * mod_value)
        image = image.resize((width, height))

        dur_sec = float(shot.get("duration_sec") or 4.0)
        raw_frames = min(max_frames_cap, max(33, int(round(dur_sec * 16))))
        num_frames = ((raw_frames - 1 + 3) // 4) * 4 + 1

        prompt = shot.get("motion_prompt") or shot["image_prompt"]
        print(f"[RENDER] {shot_id} ({width}x{height}, {num_frames}f, {steps} steps): {prompt[:80]}...")
        frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
        ).frames[0]
        export_to_video(frames, str(out_mp4), fps=16)
        print(f"[SAVED] {out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().parents[1] / "script_action_demo.json")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    render_wan22_action_shots(args.base_dir.resolve(), args.script.resolve(), steps=args.steps)
