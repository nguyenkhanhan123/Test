import json
import os
import subprocess
import sys
import urllib.request


def download_male_vrm(dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500_000:
        print("Dùng lại file VRM nam đã có:", dest_path, flush=True)
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    candidates = []

    # Tự tìm đúng đường dẫn masc_vroid.vrm trong repo madjin/vrm-samples qua GitHub API
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/madjin/vrm-samples/git/trees/master?recursive=1",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            tree = json.loads(resp.read().decode("utf-8")).get("tree", [])
        for item in tree:
            p = item.get("path", "")
            low = p.lower()
            if low.endswith(".vrm") and any(k in low for k in ("masc", "male", "orion", "avatarsample_b")):
                priority = 0 if "masc" in low else (1 if "male" in low else (2 if "orion" in low else 3))
                candidates.append((priority, f"https://raw.githubusercontent.com/madjin/vrm-samples/master/{p}"))
        candidates.sort(key=lambda x: x[0])
    except Exception as e:
        print("⚠️ Không truy vấn được GitHub API:", e, flush=True)

    urls = [u for _, u in candidates] + [
        "https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/stable/masc_vroid.vrm",
        "https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/masc_vroid.vrm",
        "https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/stable/AvatarSample_B.vrm",
    ]

    for url in urls:
        try:
            print("Đang tải nhân vật NAM VRM:", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            if os.path.getsize(dest_path) > 500_000:
                print(f"-> Tải thành công ({os.path.getsize(dest_path)/1024/1024:.2f} MB)!", flush=True)
                return
        except Exception as e:
            print("  Bỏ qua URL lỗi:", e, flush=True)

    raise RuntimeError("Không tải được file mẫu VRM nam từ GitHub.")


def main():
    theme = sys.argv[1] if len(sys.argv) > 1 else "hoa_do"
    vrm_path = "/content/Test/assets/models_3d/hoa_do_male.vrm"
    out_glb = f"/content/Test/assets/models_3d/{theme}_vrm_rigged.glb"
    out_mp4 = f"/content/Test/build_output/demo_vrm_{theme}.mp4"
    download_male_vrm(vrm_path)

    cmd = [
        "/content/blender/blender", "--background",
        "--python", "/content/Test/vrm_demo_blender.py", "--",
        vrm_path, out_glb, out_mp4, theme,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
