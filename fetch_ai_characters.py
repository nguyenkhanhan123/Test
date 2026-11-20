import io
import gc
import json
import argparse
import urllib.parse
from pathlib import Path
import numpy as np
import requests
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError:
    torch = None
    StableDiffusionPipeline = None

try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _REMBG_SESSION = rembg_new_session("u2netp")
except ImportError:
    rembg_remove = None
    _REMBG_SESSION = None

ROOT_DIR = Path(".")
CHAR_DIR = ROOT_DIR / "assets" / "characters"
BG_DIR = ROOT_DIR / "assets" / "backgrounds"

CHAR_VISUAL_THEMES = {
    "hoa_do": ("crimson red and gold", "phoenix", 202601, "young passionate male academy cadet in red and black futuristic tactical uniform"),
    "thuy_lam": ("cobalt blue and silver", "water dragon", 202602, "calm intelligent male academy cadet in blue and white tactical uniform"),
    "tho_vang": ("amber yellow and bronze", "golden lion", 202603, "sturdy determined male academy cadet in yellow and dark grey tactical uniform"),
    "phong_luc": ("emerald green and white", "sky falcon", 202604, "agile cheerful male academy cadet in green and white tactical uniform"),
    "loi_hong": ("magenta pink and violet", "thunderbird", 202605, "sharp-eyed female tech academy cadet in pink and dark tactical uniform"),
    "long_vu": ("royal gold and white", "imperial dragon", 202606, "noble white-haired male warrior in white and gold galactic armor"),
    "chi_huy": ("ice white and cyan", "aurora", 202607, "dignified female military commander in white and cyan sci-fi officer coat"),
    "mac_khieu": ("dark purple and black", "shadow demon", 202608, "menacing dark alien warlord in obsidian purple cloak and armor"),
    "quai_vat_tuan": ("charcoal grey and crimson", "dark beast", 202609, "ferocious biomechanical tokusatsu monster with glowing red eyes"),
    "phu_giap": ("dark violet and gunmetal", "shadow beetle", 202610, "armored dark alien foot-soldier monster with purple energy core"),
}


def remove_bg_rgba(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    if rembg_remove is not None:
        out_bytes = rembg_remove(raw_bytes, session=_REMBG_SESSION)
        return Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    im = img.convert("RGBA")
    arr = np.array(im)
    white_mask = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
    arr[white_mask, 3] = 0
    return Image.fromarray(arr, "RGBA")


def fetch_pollinations_image(prompt: str, width: int, height: int, seed: int) -> Image.Image | None:
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
    try:
        resp = requests.get(url, timeout=35)
        if resp.status_code == 200 and len(resp.content) > 5000:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


def build_char_prompts(char_id: str) -> dict:
    colors, motif, seed, civ_desc = CHAR_VISUAL_THEMES.get(
        char_id,
        ("silver and blue", "star", 202699, "futuristic anime academy cadet in sci-fi uniform"),
    )
    return {
        "civilian": (
            f"anime character portrait upper body, {civ_desc}, standing heroic pose, "
            f"isolated on solid plain pure white background, sharp anime cel-shaded art style, clean lines, masterpiece",
            512, 640, 600, 750, seed, True
        ),
        "armored": (
            f"tokusatsu super sentai ranger warrior, sleek metallic {colors} armor, "
            f"glowing black visor with sharp {motif} crest on helmet, heroic battle stance, "
            f"isolated on solid plain pure white background, sharp anime cel-shaded art style, masterpiece",
            512, 640, 600, 750, seed + 100, True
        ),
        "awakened": (
            f"tokusatsu super sentai warrior upgraded awakened form, heavy {colors} and obsidian armor plates "
            f"with glowing elemental {motif} energy aura and heavy shoulder pauldrons, "
            f"isolated on solid plain pure white background, dynamic anime tokusatsu art, masterpiece",
            512, 640, 600, 750, seed + 200, True
        ),
        "eye_strip": (
            f"extreme close up horizontal narrow strip of determined anime hero eyes glowing with {colors} {motif} energy, "
            f"dramatic speed lines, intense tokusatsu anime cut-in frame, masterpiece",
            768, 256, 960, 220, seed + 300, False
        ),
    }


def load_gpu_sd_pipe():
    if torch is None or StableDiffusionPipeline is None or not torch.cuda.is_available():
        return None
    print("Đang nạp model GPU Lykon/dreamshaper-8 để vẽ nhân vật & bối cảnh sắc nét...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def generate_with_pipe(pipe, prompt: str, gen_w: int, gen_h: int, seed: int, neg: str) -> Image.Image:
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=gen_w,
        height=gen_h,
        num_inference_steps=22,
        guidance_scale=7.5,
        generator=g,
    ).images[0]


def main():
    parser = argparse.ArgumentParser(description="Tạo ảnh nhân vật & bối cảnh Anime/Tokusatsu bằng GPU (hoặc Pollinations)")
    parser.add_argument("--chars", nargs="*", help="Danh sách char_id cần tạo")
    parser.add_argument("--force", action="store_true", help="Vẽ đè lại toàn bộ ảnh nhân vật & bối cảnh")
    parser.add_argument("--force-bg", action="store_true", help="Vẽ đè lại các bối cảnh bị fallback (dung lượng nhỏ < 100KB)")
    args = parser.parse_args()

    with open(ROOT_DIR / "global_registry.json", "r", encoding="utf-8") as f:
        reg = json.load(f)

    pipe = load_gpu_sd_pipe()
    neg_char = "blurry, low quality, deformed, extra limbs, text, watermark, complex background, scenery"
    neg_bg = "blurry, low quality, text, watermark, signature, characters, people"

    # 1. Tạo / nâng cấp toàn bộ bối cảnh cố định
    BG_DIR.mkdir(parents=True, exist_ok=True)
    all_bgs = dict(reg.get("backgrounds", {}))
    ep1_path = ROOT_DIR / "episodes" / "ep_001.json"
    if ep1_path.exists():
        ep1 = json.loads(ep1_path.read_text(encoding="utf-8"))
        all_bgs.update(ep1.get("custom_backgrounds", {}))

    for bg_id, bg_info in all_bgs.items():
        out_jpg = BG_DIR / f"{bg_id}_youtube.jpg"
        is_fallback = out_jpg.exists() and out_jpg.stat().st_size < 90000
        if not args.force and out_jpg.exists() and not (args.force_bg and is_fallback):
            continue
        prompt = f"{bg_info['prompt']}, {reg['style_suffix']}"
        seed = bg_info.get("seed", 1000)
        print(f"  [BG] Đang vẽ bối cảnh '{bg_id}' (seed={seed})...")
        if pipe is not None:
            im = generate_with_pipe(pipe, prompt, 960, 544, seed, neg_bg).resize((1920, 1080), Image.Resampling.LANCZOS)
        else:
            im = fetch_pollinations_image(prompt, 1920, 1080, seed)
        if im is not None:
            im.save(out_jpg, quality=94)
            print(f"    -> Đã lưu {out_jpg}")

    # 2. Tạo ảnh nhân vật & eye_strip
    char_ids = args.chars or [cid for cid in reg["characters"].keys() if cid != "narrator"]
    print(f"Đang chuẩn bị tạo ảnh AI (GPU={'ON' if pipe else 'OFF'}, rembg={'ON' if rembg_remove else 'FALLBACK'}) cho {len(char_ids)} nhân vật...")

    for cid in char_ids:
        c_dir = CHAR_DIR / cid
        c_dir.mkdir(parents=True, exist_ok=True)
        for name, (prompt, gw, gh, tw, th, seed, strip_bg) in build_char_prompts(cid).items():
            out_path = c_dir / f"{name}.png"
            min_ok = 25000 if name == "eye_strip" else 10000
            if not args.force and out_path.exists() and out_path.stat().st_size > min_ok:
                print(f"  [Skip] {cid}/{name}.png đã tồn tại")
                continue

            print(f"  [Gen] {cid}/{name}.png (seed={seed})...")
            if pipe is not None:
                im_rgb = generate_with_pipe(pipe, prompt, gw, gh, seed, neg_char)
            else:
                im_rgb = fetch_pollinations_image(prompt, tw, th, seed)

            if im_rgb is None:
                print(f"    -> Bỏ qua {cid}/{name}.png")
                continue

            if strip_bg:
                im = remove_bg_rgba(im_rgb).resize((tw, th), Image.Resampling.LANCZOS)
            else:
                im = im_rgb.convert("RGBA").resize((tw, th), Image.Resampling.LANCZOS)
            im.save(out_path)
            print(f"    -> Đã lưu {out_path}")

    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print("-> HOÀN TẤT TẠO KHO ẢNH NHÂN VẬT & BỐI CẢNH AI!")


if __name__ == "__main__":
    main()
