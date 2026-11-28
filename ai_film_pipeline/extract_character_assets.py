"""Extract transparent character sprites from cast_sheet.png & slice LoRA dataset crops."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def extract_sprite_floodfill(
    img_rgb: Image.Image,
    crop_box: tuple[int, int, int, int],
    out_path: Path,
    tol: float = 26.0,
) -> Path:
    """Extract a clean RGBA sprite by flood-filling the uniform slate background from borders."""
    sub = img_rgb.crop(crop_box).convert("RGB")
    arr = np.asarray(sub, dtype=np.float32)
    h, w, _ = arr.shape

    # Sample corner background reference
    corners = np.array([arr[2, 2], arr[2, w - 3], arr[h - 3, 2], arr[h - 3, w - 3]])
    bg = np.median(corners, axis=0)

    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    is_bg = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if dist[y, x] < tol and not is_bg[y, x]:
                is_bg[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if dist[y, x] < tol and not is_bg[y, x]:
                is_bg[y, x] = True
                queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not is_bg[ny, nx]:
                # Allow slightly softer step from neighbors to catch smooth background gradients
                if dist[ny, nx] < tol * 1.25:
                    is_bg[ny, nx] = True
                    queue.append((ny, nx))

    alpha = (~is_bg).astype(np.uint8) * 255
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.8))

    # Trim transparent margins cleanly
    rgba = sub.convert("RGBA")
    rgba.putalpha(alpha_img)
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)
    print(f"Extracted sprite {out_path.name}: {rgba.size}")
    return out_path


def build_lora_dataset_from_existing_images(base_dir: Path, trigger: str = "nv_minhkhang") -> int:
    """Create a 12-image multi-angle/multi-scale LoRA training set from existing keyframes."""
    photos_dir = base_dir / "user_inputs" / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Clear old auto-generated mk_*.png first
    for old in photos_dir.glob("mk_*.png"):
        old.unlink()
        old.with_suffix(".txt").unlink(missing_ok=True)

    cast = Image.open(base_dir / "assets" / "characters" / "cast_sheet.png").convert("RGB")
    s1 = Image.open(base_dir / "assets" / "keyframes" / "shot_001.png").convert("RGB")
    s4 = Image.open(base_dir / "assets" / "keyframes" / "shot_004.png").convert("RGB")
    s5 = Image.open(base_dir / "assets" / "keyframes" / "shot_005.png").convert("RGB")
    s6 = Image.open(base_dir / "assets" / "keyframes" / "shot_006.png").convert("RGB")

    # Curated crops focusing on Minh Khang across scale/lighting/pose variations
    crops: list[tuple[str, Image.Image, tuple[int, int, int, int], str]] = [
        ("mk_01", cast, (150, 30, 590, 768), f"{trigger}, full body standing portrait, neutral slate background, hooded dark navy trench coat, silver collar clasp, brown leather satchel"),
        ("mk_02", cast, (220, 30, 520, 420), f"{trigger}, upper body portrait facing forward, sharp dark eyes, messy black hair, hooded trench coat, silver clasp"),
        ("mk_03", cast, (265, 35, 455, 225), f"{trigger}, close-up face portrait, intense expression, messy black hair, neutral lighting"),
        ("mk_04", s1, (540, 60, 1180, 768), f"{trigger}, medium shot three-quarter view looking up at crimson sky, mossy stone balcony, dark navy trench coat, dramatic red rim lighting"),
        ("mk_05", s1, (730, 70, 1080, 410), f"{trigger}, close-up side profile looking upward, crimson light reflecting on face and black hair"),
        ("mk_06", s1, (690, 260, 1120, 720), f"{trigger}, upper torso detail, dark navy hooded trench coat, silver clasp, leather satchel strap, rolled sleeves"),
        ("mk_07", s4, (650, 15, 1310, 768), f"{trigger}, medium close-up shocked expression, warm amber glow from below, cold blue night mist background"),
        ("mk_08", s4, (680, 20, 1080, 420), f"{trigger}, close-up intense eyes and slightly open mouth, warm orange illumination, messy black hair"),
        ("mk_09", s5, (90, 40, 810, 768), f"{trigger}, dynamic side-profile action pose pointing forward, coat billowing, dark forest night"),
        ("mk_10", s5, (230, 55, 620, 430), f"{trigger}, fierce determined side profile, windswept black hair, silver collar clasp"),
        ("mk_11", s5, (90, 40, 810, 768), f"{trigger}, mirrored dynamic pose", ),
        ("mk_12", s6, (540, 430, 730, 760), f"{trigger}, rear view standing facing glowing crimson portal, navy trench coat and brown satchel across back"),
    ]

    for name, src, box, caption in crops:
        crop_im = src.crop(box)
        if name == "mk_11":
            crop_im = crop_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            caption = f"{trigger}, dynamic action pose pointing left, coat billowing, dramatic red embers"
        # Upscale small crops nicely to >=512px short side for crisp LoRA bucketing
        min_side = min(crop_im.size)
        if min_side < 512:
            scale = 512.0 / min_side
            crop_im = crop_im.resize(
                (int(round(crop_im.width * scale)), int(round(crop_im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        out_png = photos_dir / f"{name}.png"
        crop_im.save(out_png)
        out_png.with_suffix(".txt").write_text(caption, encoding="utf-8")

    print(f"Built {len(crops)} LoRA training images + captions in {photos_dir}")
    return len(crops)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cast = Image.open(base_dir / "assets" / "characters" / "cast_sheet.png").convert("RGB")
    sprites_dir = base_dir / "assets" / "sprites"
    # Left half: Minh Khang (x=0..688), Right half: Linh Chi (x=688..1376)
    extract_sprite_floodfill(cast, (80, 15, 640, 768), sprites_dir / "minh_khang.png")
    extract_sprite_floodfill(cast, (820, 50, 1250, 768), sprites_dir / "linh_chi.png")
    build_lora_dataset_from_existing_images(base_dir)


if __name__ == "__main__":
    main()
