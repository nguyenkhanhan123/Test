import io
import json
import argparse
import urllib.parse
from pathlib import Path
import numpy as np
import requests
from PIL import Image

try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _REMBG_SESSION = rembg_new_session("u2netp")
except ImportError:
    rembg_remove = None
    _REMBG_SESSION = None

ROOT_DIR = Path(".")
CHAR_DIR = ROOT_DIR / "assets" / "characters"

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


def remove_bg_rgba(img_bytes: bytes) -> Image.Image:
    if rembg_remove is not None:
        out_bytes = rembg_remove(img_bytes, session=_REMBG_SESSION)
        return Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    arr = np.array(im)
    # Fallback: xóa nền trắng/sáng đơn sắc ở biên ảnh
    white_mask = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
    arr[white_mask, 3] = 0
    return Image.fromarray(arr, "RGBA")


def fetch_pollinations_image(prompt: str, width: int, height: int, seed: int) -> bytes | None:
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
    try:
        resp = requests.get(url, timeout=45)
        if resp.status_code == 200 and len(resp.content) > 5000:
            return resp.content
    except Exception as exc:
        print(f"    [Warning] Pollinations error: {exc}")
    return None


def build_prompts(char_id: str) -> dict:
    colors, motif, seed, civ_desc = CHAR_VISUAL_THEMES.get(
        char_id,
        ("silver and blue", "star", 202699, "futuristic anime academy cadet in sci-fi uniform"),
    )
    return {
        "civilian": (
            f"full body anime character portrait of {civ_desc}, standing heroic pose, "
            f"isolated on solid plain pure white background, sharp anime cel-shaded art style, clean lines, 8k",
            600, 750, seed, True
        ),
        "armored": (
            f"full body character design of a tokusatsu super sentai warrior, sleek metallic {colors} armor, "
            f"glowing black visor with sharp {motif} crest on helmet, heroic battle stance, "
            f"isolated on solid plain pure white background, sharp anime cel-shaded art style, 8k",
            600, 750, seed + 100, True
        ),
        "awakened": (
            f"full body tokusatsu super sentai warrior upgraded awakened form, heavy {colors} and obsidian armor plates "
            f"with glowing elemental {motif} energy aura and heavy shoulder pauldrons, "
            f"isolated on solid plain pure white background, dynamic anime tokusatsu art, 8k",
            600, 750, seed + 200, True
        ),
        "eye_strip": (
            f"extreme close up horizontal narrow strip of determined anime hero eyes glowing with {colors} {motif} energy, "
            f"dramatic speed lines, intense tokusatsu anime cut-in frame, 8k",
            960, 220, seed + 300, False
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Tải ảnh nhân vật Anime/Tokusatsu từ Pollinations AI + tách nền rembg")
    parser.add_argument("--chars", nargs="*", help="Danh sách char_id cần tải (mặc định: toàn bộ trong global_registry.json)")
    parser.add_argument("--force", action="store_true", help="Tải đè lại dù file đã tồn tại")
    args = parser.parse_args()

    with open(ROOT_DIR / "global_registry.json", "r", encoding="utf-8") as f:
        reg = json.load(f)

    char_ids = args.chars or [cid for cid in reg["characters"].keys() if cid != "narrator"]
    print(f"Đang chuẩn bị tạo ảnh AI (rembg={'ON' if rembg_remove else 'FALLBACK'}) cho {len(char_ids)} nhân vật...")

    for cid in char_ids:
        c_dir = CHAR_DIR / cid
        c_dir.mkdir(parents=True, exist_ok=True)
        prompts = build_prompts(cid)
        for name, (prompt, w, h, seed, strip_bg) in prompts.items():
            out_path = c_dir / f"{name}.png"
            if not args.force and name != "eye_strip" and out_path.exists() and out_path.stat().st_size > 10000:
                print(f"  [Skip] {cid}/{name}.png đã tồn tại")
                continue
            if not args.force and name == "eye_strip" and out_path.exists() and out_path.stat().st_size > 25000:
                print(f"  [Skip] {cid}/eye_strip.png đã tồn tại")
                continue

            print(f"  [Fetch] {cid}/{name}.png (seed={seed})...")
            raw = fetch_pollinations_image(prompt, w, h, seed)
            if not raw:
                print(f"    -> Bỏ qua {cid}/{name}.png (không tải được)")
                continue

            if strip_bg:
                im = remove_bg_rgba(raw).resize((w, h), Image.Resampling.LANCZOS)
            else:
                im = Image.open(io.BytesIO(raw)).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
            im.save(out_path)
            print(f"    -> Đã lưu {out_path}")

    print("-> HOÀN TẤT TẠO KHO ẢNH NHÂN VẬT AI!")


if __name__ == "__main__":
    main()
