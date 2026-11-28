"""Colab Pro (L4/A100) Step 5 Template: Batch Wan 2.2 I2V via ComfyUI API + Accelerators.

Integrates the 3 required accelerators from the report:
1. SageAttention (INT8 QK^T)
2. TeaCache (threshold=0.40 wide/action, lowered to 0.25 on character close-ups)
3. LightX2V 4-8 step distill LoRA + FP8 weights

Only renders shots marked ``shot_type == "video_ai"`` (~30% of episode) and reads
exact ``duration_sec`` computed by Step 3 (Audio-First) so clips never drift.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/ai_film_series")
COMFY_URL = "http://127.0.0.1:8188"


def submit_comfy_prompt(workflow: dict) -> str:
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["prompt_id"]


def wait_for_comfy_completion(prompt_id: str, poll_sec: float = 3.0) -> dict:
    while True:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as resp:
            hist = json.loads(resp.read().decode("utf-8"))
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(poll_sec)


def build_wan22_i2v_workflow(
    image_filename: str,
    motion_prompt: str,
    n_frames: int,
    width: int,
    height: int,
    character_lora: str | None,
    teacache_thresh: float,
) -> dict:
    """Return a ComfyUI API workflow graph skeleton for Wan 2.2 I2V + LightX2V + TeaCache."""
    # NOTE: Export your verified ComfyUI API JSON template from ComfyUI once and
    # parameterize node inputs cleanly here for 100% deterministic batch rendering.
    return {
        "meta": {
            "model_high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "model_low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "distill_lora": "Wan2.2_LightX2V_I2V_4step.safetensors",
            "character_lora": character_lora,
            "sage_attention": "sageattn_qk_int8_pv_fp16_cuda",
            "teacache_rel_l1_thresh": teacache_thresh,
            "steps": 6,
            "cfg": 1.0,
            "width": width,
            "height": height,
            "length_frames": n_frames,  # Wan models expect 4k+1 frames (e.g. 121 frames = 5.0s @ 24fps)
            "image": image_filename,
            "positive_prompt": motion_prompt,
        }
    }


def batch_render_video_ai_shots(script_path: Path, out_dir: Path) -> None:
    script = json.loads(script_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(script.get("fps", 24))
    width = int(script.get("width", 1280))
    height = int(script.get("height", 720))

    ai_shots = [s for s in script["shots"] if s["shot_type"] == "video_ai"]
    print(f"Found {len(ai_shots)} video_ai shots to render (skipping static Ken Burns shots).")

    for shot in ai_shots:
        shot_id = shot["shot_id"]
        out_clip = out_dir / f"{shot_id}.mp4"
        if out_clip.exists():
            print(f"[RESUME SKIP] {out_clip} already exists on Drive.")
            continue

        dur = float(shot.get("duration_sec") or 5.0)
        # Wan VAE temporal compression uses 4*k + 1 frames; generate >= target and let Step 6 trim cleanly
        raw_frames = int(round(dur * fps))
        wan_frames = ((max(17, raw_frames) - 1 + 3) // 4) * 4 + 1
        # Lower TeaCache threshold on close-up character shots to protect facial identity
        teacache_thresh = 0.25 if shot.get("character_lora") else 0.40

        workflow = build_wan22_i2v_workflow(
            image_filename=f"{shot_id}.png",
            motion_prompt=shot.get("motion_prompt") or shot["image_prompt"],
            n_frames=wan_frames,
            width=width,
            height=height,
            character_lora=shot.get("character_lora"),
            teacache_thresh=teacache_thresh,
        )
        print(f"[QUEUE] {shot_id}: frames={wan_frames}, teacache={teacache_thresh}, lora={shot.get('character_lora')}")
        # Replace with submit_comfy_prompt(workflow) once ComfyUI headless server is started on port 8188
        (out_dir / f"{shot_id}_comfy_job.json").write_text(
            json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    try:
        batch_render_video_ai_shots(
            script_path=DRIVE_ROOT / "episode_01/script.json",
            out_dir=DRIVE_ROOT / "episode_01/assets/video_ai",
        )
    finally:
        try:
            from google.colab import runtime  # type: ignore

            runtime.unassign()
        except ImportError:
            pass
