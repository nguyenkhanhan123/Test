"""Colab T4/L4 Cell 3 Script: Generate Action Keyframes from Trained SDXL LoRA.

Uses ``variant="fp16"``, ``low_cpu_mem_usage=True``, and ``enable_model_cpu_offload()``
so SDXL fits well inside Colab T4's 12.7 GB system RAM without triggering Linux OOM.
"""

from __future__ import annotations

import gc
import glob
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline


def generate_action_keyframes(base_dir: Path) -> None:
    lora_files = sorted(glob.glob(str(base_dir / "loras" / "nv_minhkhang_lora_v1" / "*.safetensors")))
    if not lora_files:
        raise RuntimeError("Chưa thấy file LoRA .safetensors trong loras/nv_minhkhang_lora_v1/!")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    dtype = torch.bfloat16 if (vram_gb >= 20.0 and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[Cell 3] Loading SDXL (variant=fp16, low_cpu_mem) + LoRA ({Path(lora_files[-1]).name})...")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    pipe.load_lora_weights(str(Path(lora_files[-1]).parent), weight_name=Path(lora_files[-1]).name)

    prompts = {
        "shot_103": (
            "nv_minhkhang, dynamic martial arts forward dash slash, glowing crimson energy arc, "
            "rain sparks, dark navy hooded trench coat, anime film still, wide 16:9"
        ),
        "shot_104": (
            "nv_minhkhang, high-speed mid-air counter strike against glowing red portal energy, "
            "dramatic rim lighting, dark navy hooded trench coat, anime film still, wide 16:9"
        ),
    }
    neg_prompt = "blurry, deformed hands, extra limbs, text, watermark, subtitle, low quality"

    out_dir = base_dir / "assets" / "keyframes_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    sc_path = base_dir / "script_action_demo.json"
    sc = json.loads(sc_path.read_text(encoding="utf-8"))

    for shot in sc["shots"]:
        sid = shot["shot_id"]
        if sid in prompts:
            img = pipe(
                prompts[sid],
                negative_prompt=neg_prompt,
                width=1024,
                height=576,
                num_inference_steps=26,
                guidance_scale=7.0,
            ).images[0]
            out_img = out_dir / f"{sid}.png"
            img.save(out_img)
            shot["keyframe_path"] = f"assets/keyframes_lora/{sid}.png"
            print(f"[Cell 3] Saved LoRA keyframe: {out_img}")

    sc_path.write_text(json.dumps(sc, indent=2, ensure_ascii=False), encoding="utf-8")
    del pipe, vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_action_keyframes(Path(__file__).resolve().parents[1])
