import shutil
import subprocess
from pathlib import Path
import torch
from huggingface_hub import snapshot_download


def main():
    if not torch.cuda.is_available():
        print("⚠️ Không có GPU CUDA — bỏ qua bước LivePortrait.")
        return

    subprocess.run(
        ["pip", "install", "-q", "pykalman", "tyro", "pyyaml", "opencv-python-headless", "scipy", "scikit-image", "rich", "albumentations", "onnxruntime-gpu", "insightface"],
        check=True,
    )

    lp_repo = Path("_tmp_LivePortrait")
    if not (lp_repo / "pretrained_weights").exists():
        shutil.rmtree(lp_repo, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/KwaiVGI/LivePortrait", str(lp_repo)], check=True)
        snapshot_download(
            repo_id="KwaiVGI/LivePortrait",
            local_dir=str(lp_repo / "pretrained_weights"),
            ignore_patterns=["*.git*", "README.md", "docs*"],
        )

    driving = (lp_repo / "assets/examples/driving/d0.mp4").resolve()
    char_root = Path("assets/characters")
    for c_dir in sorted(char_root.iterdir()):
        src_img = c_dir / "civilian.png"
        target_mp4 = c_dir / "liveportrait_civilian.mp4"
        if not src_img.exists() or target_mp4.exists():
            continue
        tmp_out = (lp_repo / "out_tmp" / c_dir.name).resolve()
        tmp_out.mkdir(parents=True, exist_ok=True)
        print("Animating LivePortrait:", c_dir.name, flush=True)
        res = subprocess.run(
            [
                "python", "inference.py",
                "-s", str(src_img.resolve()),
                "-d", str(driving),
                "-o", str(tmp_out),
            ],
            cwd=str(lp_repo),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"⚠️ LivePortrait lỗi ở {c_dir.name}, bỏ qua:", res.stderr[-500:], flush=True)
            continue
        for mp4 in tmp_out.glob("*.mp4"):
            if "concat" not in mp4.name:
                shutil.copy2(mp4, target_mp4)
                print("Saved:", target_mp4, flush=True)
                break


if __name__ == "__main__":
    main()
