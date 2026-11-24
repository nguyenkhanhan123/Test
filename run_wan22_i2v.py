import gc
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def make_cinematic_keyframe(apose_path, out_path, width=1280, height=704):
    """Đặt nhân vật Hỏa Đỏ từ ảnh A-Pose vào bối cảnh đấu trường Super Sentai 16:9 điện ảnh."""
    src = Image.open(apose_path).convert("RGBA")
    try:
        from rembg import remove
        fg = remove(src)
    except Exception:
        # Fallback tách nền trắng đơn giản nếu rembg chưa có
        arr = np.array(src)
        white = (arr[..., 0] > 240) & (arr[..., 1] > 240) & (arr[..., 2] > 240)
        arr[white, 3] = 0
        fg = Image.fromarray(arr, "RGBA")

    bbox = fg.getbbox()
    if bbox:
        fg = fg.crop(bbox)

    # Tạo nền đấu trường Tokusatsu tối có ánh lửa đỏ từ dưới hắt lên
    y_grad = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x_grad = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    bg_arr = np.zeros((height, width, 3), dtype=np.float32)
    # Bầu trời đêm xanh-xám đậm chuyển xuống mặt đất tối
    bg_arr[..., 0] = 16 + 38 * np.exp(-((x_grad * 1.4) ** 2)) * np.exp(-(((y_grad - 0.82) * 3.5) ** 2))
    bg_arr[..., 1] = 18 + 10 * (1.0 - y_grad)
    bg_arr[..., 2] = 28 + 18 * (1.0 - y_grad)
    # Vệt sáng lửa đỏ hắt sau lưng nhân vật (Rim glow)
    glow = np.exp(-((x_grad * 2.4) ** 2)) * np.exp(-(((y_grad - 0.55) * 2.2) ** 2))
    bg_arr[..., 0] += 95 * glow
    bg_arr[..., 1] += 18 * glow
    bg_arr[..., 2] += 12 * glow
    bg = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8), "RGB")

    # Vẽ đường chân trời & vài tia sáng lửa nhẹ
    draw = ImageDraw.Draw(bg)
    horizon_y = int(height * 0.78)
    draw.rectangle([0, horizon_y, width, height], fill=(14, 15, 20))
    draw.line([(0, horizon_y), (width, horizon_y)], fill=(120, 32, 24), width=2)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=4))

    # Dán nhân vật Hỏa Đỏ vào trung tâm khung hình (cao ~90% khung)
    target_h = int(height * 0.92)
    scale = target_h / max(1, fg.height)
    nw, nh = max(1, int(round(fg.width * scale))), target_h
    fg_res = fg.resize((nw, nh), Image.Resampling.LANCZOS)

    # Bóng đổ mềm dưới chân
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    cx = width // 2
    foot_y = int(height * 0.96)
    sdraw.ellipse([cx - nw // 2, foot_y - 16, cx + nw // 2, foot_y + 14], fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    bg = Image.alpha_composite(bg.convert("RGBA"), shadow)

    paste_x = (width - nw) // 2
    paste_y = max(10, foot_y - nh)
    bg.paste(fg_res, (paste_x, paste_y), fg_res)
    final_rgb = bg.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_rgb.save(out_path)
    print(f"Đã tạo ảnh khung hình điện ảnh 16:9: {out_path} ({width}x{height})", flush=True)
    return final_rgb


def main():
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    apose_png = "/content/Test/assets/apose_inputs/hoa_do_apose.png"
    keyframe_png = "/content/Test/build_output/hoa_do_wan22_keyframe.png"
    out_mp4 = "/content/Test/build_output/demo_wan22_hoa_do.mp4"
    os.makedirs("/content/Test/build_output", exist_ok=True)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU VRAM: {vram_gb:.1f} GB | dtype={dtype}", flush=True)

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    print(f"1/3 Đang nạp model {model_id}...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)

    if vram_gb < 22:
        print("-> GPU <22GB (T4): bật Group Offloading theo khối để giảm VRAM xuống ~6GB...", flush=True)
        from diffusers.hooks.group_offloading import apply_group_offloading
        onload_dev = torch.device("cuda")
        offload_dev = torch.device("cpu")
        apply_group_offloading(
            pipe.text_encoder,
            onload_device=onload_dev,
            offload_device=offload_dev,
            offload_type="block_level",
            num_blocks_per_group=4,
            low_cpu_mem_usage=True,
        )
        pipe.transformer.enable_group_offload(
            onload_device=onload_dev,
            offload_device=offload_dev,
            offload_type="block_level",
            num_blocks_per_group=2,
            low_cpu_mem_usage=True,
        )
        pipe.vae.to(onload_dev)
    else:
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Tính kích thước chuẩn bội số của Wan2.2 VAE + patch_size
    max_area = (1280 * 704) if vram_gb >= 22 else (832 * 480)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    aspect = 704 / 1280  # 16:9 widescreen
    height = int(round(np.sqrt(max_area * aspect))) // mod_value * mod_value
    width = int(round(np.sqrt(max_area / aspect))) // mod_value * mod_value
    num_frames = 81 if vram_gb >= 22 else 49  # 81 frames (~3.4s @ 24fps) trên A100/L4, 49 frames (~2s) trên T4
    steps = 35 if vram_gb >= 22 else 28

    print(f"2/3 Đang dựng ảnh gốc 16:9 ({width}x{height})...", flush=True)
    init_img = make_cinematic_keyframe(apose_png, keyframe_png, width=width, height=height)

    prompt = (
        "Cinematic live-action Japanese Tokusatsu Super Sentai movie scene. "
        "The heroic young Vietnamese warrior in the red and black A.E.G.I.S tactical jacket shifts smoothly "
        "from standing into a powerful wide martial-arts fighting stance, clenching both hands tightly into fists. "
        "Fiery crimson sparks and glowing embers swirl around his fists as he throws a sharp, decisive forward punch "
        "with intense determination on his face. Dramatic low-angle cinematic camera slowly pushes in, "
        "sharp focus, realistic 3D CGI lighting, fiery red rim light, ultra detailed, 24fps film look."
    )
    negative_prompt = (
        "blurry, distorted face, melted face, extra fingers, deformed hands, bad anatomy, "
        "static, low quality, worst quality, text, watermark, cartoon, flat lighting, flickering"
    )

    print(f"3/3 Đang render video AI Wan2.2 ({width}x{height}, {num_frames} frames, {steps} steps)...", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(202605)
    frames = pipe(
        image=init_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=5.0,
        num_inference_steps=steps,
        generator=generator,
    ).frames[0]

    export_to_video(frames, out_mp4, fps=24)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"-> Hoàn tất video AI Wan2.2: {out_mp4} ({os.path.getsize(out_mp4)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
