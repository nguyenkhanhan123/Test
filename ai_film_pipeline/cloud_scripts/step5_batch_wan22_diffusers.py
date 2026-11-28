"""Colab Pro (L4/T4) Step 5 Runnable Script: Wan 2.2 TI2V-5B Action Video Generator.

Uses Hugging Face Diffusers ``WanImageToVideoPipeline`` with
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` so you can generate real AI action video
clips directly inside Google Colab without setting up ComfyUI nodes manually.
Supports resume-from-Drive and automatic duration-matched frame counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def render_wan22_action_shots(
    base_dir: Path,
    script_path: Path,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    steps: int = 30,
    max_area: int = 480 * 832,
) -> None:
    script = json.loads(script_path.read_text(encoding="utf-8"))
    out_dir = base_dir / "assets" / "video_ai"
    out_dir.mkdir(parents=True, exist_ok=True)

    ai_shots = [s for s in script["shots"] if s["shot_type"] == "video_ai"]
    pending = [s for s in ai_shots if not (out_dir / f"{s['shot_id']}.mp4").exists()]
    if not pending:
        print("All video_ai shots already rendered.")
        return

    dtype = pick_dtype()
    print(f"Loading {model_id} ({dtype}) for {len(pending)} action shots...")
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
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

        dur_sec = float(shot.get("duration_sec") or 4.5)
        # Wan 2.2 TI2V-5B runs naturally at 24fps (or 16fps); cap at 121 frames (~5s @ 24fps) to fit VRAM
        raw_frames = min(121, max(49, int(round(dur_sec * 24))))
        num_frames = ((raw_frames - 1 + 3) // 4) * 4 + 1

        prompt = shot.get("motion_prompt") or shot["image_prompt"]
        print(f"[RENDER] {shot_id} ({width}x{height}, {num_frames} frames, {steps} steps): {prompt[:80]}...")
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
        export_to_video(frames, str(out_mp4), fps=24)
        print(f"[SAVED] {out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().parents[1] / "script_action_demo.json")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    render_wan22_action_shots(args.base_dir.resolve(), args.script.resolve(), steps=args.steps)
