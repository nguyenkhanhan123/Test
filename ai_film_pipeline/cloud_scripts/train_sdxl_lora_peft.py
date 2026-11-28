"""Colab T4 (15GB) & L4 (24GB) SDXL Character LoRA Trainer via Diffusers + PEFT.

Pre-caches VAE latents and dual CLIP text embeddings before training UNet LoRA
with 8-bit AdamW and gradient checkpointing. Avoids ostris/ai-toolkit's pinned
scipy/numpy build failures on Colab Python 3.12 while producing standard SDXL LoRA
weights loadable via ``pipe.load_lora_weights(...)``.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _encode_prompt(
    prompt: str,
    tok1: AutoTokenizer,
    tok2: AutoTokenizer,
    te1: CLIPTextModel,
    te2: CLIPTextModelWithProjection,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    in1 = tok1(prompt, padding="max_length", max_length=tok1.model_max_length, truncation=True, return_tensors="pt").to(device)
    in2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = te1(in1.input_ids, output_hidden_states=True)
        out2 = te2(in2.input_ids, output_hidden_states=True)
        prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
        pooled = out2[0]
    return prompt_embeds.cpu(), pooled.cpu()


def train_sdxl_lora(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
    steps: int = 360,
    resolution: int = 768,
    rank: int = 16,
    lr: float = 1e-4,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No training images found in {dataset_dir}")
    print(f"[LoRA] Dataset: {len(images)} images | res={resolution} | steps={steps} | dtype={weight_dtype}")

    # 1. Cache text embeddings then free text encoders from VRAM
    tok1 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    tok2 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=weight_dtype).to(device)
    te1.eval()
    te2.eval()

    # 2. Cache VAE latents then free VAE from VRAM
    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=weight_dtype).to(device)
    vae.eval()

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    cached_samples: list[dict[str, torch.Tensor]] = []
    add_time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], dtype=weight_dtype)

    for img_path in images:
        cap_path = img_path.with_suffix(".txt")
        caption = cap_path.read_text(encoding="utf-8").strip() if cap_path.exists() else "anime character portrait"
        p_emb, pooled = _encode_prompt(caption, tok1, tok2, te1, te2, device)
        with Image.open(img_path) as im:
            px = tfm(im.convert("RGB")).unsqueeze(0).to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        cached_samples.append({
            "latents": latents.cpu(),
            "prompt_embeds": p_emb,
            "pooled_embeds": pooled,
        })

    del te1, te2, vae, tok1, tok2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Train UNet LoRA only (fits comfortably in 15GB Tesla T4)
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=weight_dtype).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    trainable = [p for p in unet.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb  # type: ignore

        optimizer = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=1e-2)
        print("[LoRA] Using bitsandbytes AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
        print("[LoRA] Using torch.optim.AdamW")

    scaler = torch.amp.GradScaler("cuda", enabled=(weight_dtype == torch.float16))
    unet.train()
    grad_accum = 2
    pbar = tqdm(range(steps), desc="Training SDXL LoRA")
    n_samples = len(cached_samples)

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            idx = int(torch.randint(0, n_samples, (1,)).item())
            sample = cached_samples[idx]
            lat = sample["latents"].to(device=device, dtype=weight_dtype)
            p_emb = sample["prompt_embeds"].to(device=device, dtype=weight_dtype)
            pooled = sample["pooled_embeds"].to(device=device, dtype=weight_dtype)
            t_ids = add_time_ids.to(device=device, dtype=weight_dtype)

            noise = torch.randn_like(lat)
            ts = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_lat = scheduler.add_noise(lat, noise, ts)

            with torch.amp.autocast("cuda", dtype=weight_dtype):
                pred = unet(
                    noisy_lat,
                    ts,
                    encoder_hidden_states=p_emb,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": t_ids},
                ).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean") / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += float(loss.detach().item())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            pbar.set_postfix(loss=f"{step_loss:.4f}")

    # Save weights in Diffusers SDXL LoRA format
    from diffusers import StableDiffusionXLPipeline

    unet_lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_sd,
    )
    out_file = output_dir / "pytorch_lora_weights.safetensors"
    print(f"[LoRA] Saved trained weights: {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()
    train_sdxl_lora(args.dataset.resolve(), args.output.resolve(), steps=args.steps, resolution=args.resolution)
