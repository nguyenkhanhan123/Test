"""Colab L4 (24GB/52GB RAM) & T4 (15GB/12.7GB RAM) Step 5 Video AI Generator.

- On L4 / A100 (>=20GB VRAM and >=24GB system RAM): runs ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``.
- On free/constrained T4 (12.7GB system RAM): automatically switches to lightweight
  ``Lightricks/LTX-Video`` (2B) with 4-bit NF4 quantized T5 text encoder so it never
  crashes Colab's 12.7 GB RAM limit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from diffusers.utils import export_to_video, load_image


def get_system_ram_gb() -> float:
    if hasattr(os, "sysconf"):
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    return 16.0


def get_hardware_tier() -> tuple[str, float, float]:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    ram_gb = get_system_ram_gb()
    if vram_gb >= 20.0 and ram_gb >= 24.0 and torch.cuda.is_bf16_supported():
        return "wan22_l4", vram_gb, ram_gb
    return "ltx_t4_lowram", vram_gb, ram_gb


def _render_with_wan22_l4(base_dir: Path, pending: list[dict], steps: int) -> None:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    out_dir = base_dir / "assets" / "video_ai"
    print(f"[Cell 5] High-RAM GPU detected -> loading {model_id} (bf16)...")

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    max_area = 480 * 832
    neg_prompt = (
        "static frame, frozen body, slow motion drift, deformed hands, extra limbs, "
        "blurry face, low quality, text, watermark, subtitle"
    )

    for shot in pending:
        shot_id = shot["shot_id"]
        image = load_image(str(base_dir / shot["keyframe_path"]))
        aspect = image.height / image.width
        height = int(round(np.sqrt(max_area * aspect)) // mod_value * mod_value)
        width = int(round(np.sqrt(max_area / aspect)) // mod_value * mod_value)
        image = image.resize((width, height))

        dur_sec = float(shot.get("duration_sec") or 4.0)
        raw_frames = min(81, max(41, int(round(dur_sec * 16))))
        num_frames = ((raw_frames - 1 + 3) // 4) * 4 + 1
        prompt = shot.get("motion_prompt") or shot["image_prompt"]

        print(f"[Wan2.2] {shot_id} ({width}x{height}, {num_frames}f, {steps} steps)...")
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
        out_mp4 = out_dir / f"{shot_id}.mp4"
        export_to_video(frames, str(out_mp4), fps=16)
        print(f"[SAVED] {out_mp4}")


def _render_with_ltx_t4(base_dir: Path, pending: list[dict], steps: int) -> None:
    from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel
    from transformers import BitsAndBytesConfig, T5EncoderModel

    model_id = "Lightricks/LTX-Video"
    out_dir = base_dir / "assets" / "video_ai"
    print(f"[Cell 5] T4 / 12.7GB RAM mode -> loading {model_id} with 4-bit T5...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe = LTXImageToVideoPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    neg_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, static frame"
    for shot in pending:
        shot_id = shot["shot_id"]
        image = load_image(str(base_dir / shot["keyframe_path"])).resize((704, 384))
        prompt = shot.get("motion_prompt") or shot["image_prompt"]
        # LTX-Video requires 8*k + 1 frames (49 frames @ 24fps = 2.04s, fast & safe for 12.7GB RAM)
        num_frames = 49

        print(f"[LTX-Video] {shot_id} (704x384, {num_frames}f, {steps} steps)...")
        frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=704,
            height=384,
            num_frames=num_frames,
            num_inference_steps=steps,
        ).frames[0]
        out_mp4 = out_dir / f"{shot_id}.mp4"
        export_to_video(frames, str(out_mp4), fps=24)
        print(f"[SAVED] {out_mp4}")

    del pipe, transformer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()


def render_wan22_action_shots(
    base_dir: Path,
    script_path: Path,
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

    tier, vram_gb, ram_gb = get_hardware_tier()
    print(f"[Cell 5] Hardware: VRAM={vram_gb:.1f}GB, RAM={ram_gb:.1f}GB -> mode={tier}")
    if tier == "wan22_l4":
        _render_with_wan22_l4(base_dir, pending, steps)
    else:
        _render_with_ltx_t4(base_dir, pending, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().parents[1] / "script_action_demo.json")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    render_wan22_action_shots(args.base_dir.resolve(), args.script.resolve(), steps=args.steps)
