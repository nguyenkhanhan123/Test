"""Kaggle Free (P100/T4) / Colab L4 Step 4 Template: Batch Keyframe Generation.

Reads ``script.json``, resumes cleanly if keyframes already exist on Drive/output,
and adapts dtype automatically for older Kaggle P100/T4 GPUs (which lack native FP8/BF16).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def pick_optimal_dtype() -> torch.dtype:
    """Use bfloat16 on Ada/Ampere (L4/A100); fall back to float16 on T4/P100."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def batch_generate_keyframes(
    script_path: Path,
    out_dir: Path,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    steps: int = 4,
    seed: int = 202601,
) -> None:
    from diffusers import DiffusionPipeline  # imported lazily for cloud environments

    script = json.loads(script_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = pick_optimal_dtype()
    print(f"Loading {model_id} in {dtype}...")

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()

    prefix = script.get("style_prompt_prefix", "")
    width = int(script.get("width", 1280))
    height = int(script.get("height", 720))

    for idx, shot in enumerate(script["shots"]):
        shot_id = shot["shot_id"]
        out_file = out_dir / f"{shot_id}.png"
        if out_file.exists():
            print(f"[SKIP] {shot_id} already exists.")
            continue

        prompt = f"{prefix}, {shot['image_prompt']}"
        generator = torch.Generator(device="cpu").manual_seed(seed + idx)
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        image.save(out_file)
        print(f"[OK] Saved keyframe {out_file}")


if __name__ == "__main__":
    batch_generate_keyframes(
        script_path=Path("/kaggle/input/episode-01/script.json"),
        out_dir=Path("/kaggle/working/keyframes"),
    )
