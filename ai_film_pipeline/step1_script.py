"""Step 1: Generate and validate the canonical pilot ``script.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

from schema import EpisodeScript, Shot, ShotType, VoiceSpec


def build_pilot_episode() -> EpisodeScript:
    """Create the 6-shot MVP pilot episode expanding the report's sample hook."""
    characters = {
        "narrator": VoiceSpec(
            engine="edge_tts",
            voice_id="vi-VN-NamMinhNeural",
            rate="-4%",
            pitch="-3Hz",
        ),
        "minh_khang": VoiceSpec(
            engine="edge_tts",
            voice_id="vi-VN-NamMinhNeural",
            rate="+3%",
            pitch="+4Hz",
        ),
        "linh_chi": VoiceSpec(
            engine="edge_tts",
            voice_id="vi-VN-HoaiMyNeural",
            rate="+2%",
            pitch="+1Hz",
        ),
    }

    shots = [
        Shot(
            shot_id="shot_001",
            character="minh_khang",
            character_lora="minh_khang_v1.safetensors",
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=characters["minh_khang"],
            shot_type=ShotType.VIDEO_AI,
            dialogue="Bầu trời đêm nay đột nhiên đỏ rực một cách kỳ lạ.",
            image_prompt=(
                "medium shot of Minh Khang standing alone on a damp stone lookout balcony at midnight, "
                "looking up with alert eyes toward a surreal glowing crimson night sky, red storm clouds "
                "illuminating his dark navy hooded trench coat and silver collar clasp, cool blue town lights below"
            ),
            motion_prompt=(
                "camera slowly tilts up, wind blowing Minh Khang's messy black hair and trench coat lapels, "
                "dramatic crimson clouds churning overhead, subtle breathing and blinking"
            ),
            sfx_cue="wind_ominous",
            lead_in_sec=0.35,
            tail_pad_sec=0.50,
            keyframe_path="assets/keyframes/shot_001.png",
        ),
        Shot(
            shot_id="shot_002",
            character="narrator",
            character_lora=None,
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=characters["narrator"],
            shot_type=ShotType.KEN_BURNS_ZOOM_IN,
            dialogue="Không ai trong thị trấn biết rằng đó là điềm báo của một kỷ nguyên mới.",
            image_prompt=(
                "wide aerial view of a quiet misty mountain town in Vietnam at midnight under a surreal "
                "glowing crimson red sky, traditional tiled roofs and warm golden street lanterns wrapped in "
                "low mountain fog, surrounding dark pine ridges"
            ),
            motion_prompt=None,
            sfx_cue="none",
            lead_in_sec=0.25,
            tail_pad_sec=0.45,
            keyframe_path="assets/keyframes/shot_002.png",
        ),
        Shot(
            shot_id="shot_003",
            character="narrator",
            character_lora=None,
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=characters["narrator"],
            shot_type=ShotType.KEN_BURNS_PAN_RIGHT,
            dialogue="Chỉ vài giây sau, một vệt sáng chói lòa xé toạc màn đêm rồi lao thẳng xuống khu rừng phía Bắc.",
            image_prompt=(
                "dramatic wide panoramic shot of a blinding crimson-white meteor streak tearing diagonally across "
                "the dark red-tinted night sky and plunging toward a deep misty pine forest ridge beyond the "
                "mountain town, volumetric light trail and distant shockwave glow"
            ),
            motion_prompt=None,
            sfx_cue="meteor_impact",
            lead_in_sec=0.25,
            tail_pad_sec=0.50,
            keyframe_path="assets/keyframes/shot_003.png",
        ),
        Shot(
            shot_id="shot_004",
            character="linh_chi",
            character_lora="linh_chi_v1.safetensors",
            voice_ref="edge_tts:vi-VN-HoaiMyNeural",
            voice_spec=characters["linh_chi"],
            shot_type=ShotType.KEN_BURNS_ZOOM_OUT,
            dialogue="Khang nhìn xem! Chiếc la bàn cổ tự nhiên xoay loạn xạ rồi phát sáng!",
            image_prompt=(
                "medium close-up of Linh Chi wearing her olive-beige field jacket and red hairpin standing next to "
                "Minh Khang on a misty cobblestone alley, holding up an antique brass compass glowing with bright "
                "amber-crimson runes that cast warm light onto both of their shocked faces"
            ),
            motion_prompt=None,
            sfx_cue="compass_chime",
            lead_in_sec=0.25,
            tail_pad_sec=0.45,
            keyframe_path="assets/keyframes/shot_004.png",
        ),
        Shot(
            shot_id="shot_005",
            character="minh_khang",
            character_lora="minh_khang_v1.safetensors",
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=characters["minh_khang"],
            shot_type=ShotType.VIDEO_AI,
            dialogue="Thứ vừa rơi xuống không phải thiên thạch đâu. Đi thôi, trước khi quá muộn!",
            image_prompt=(
                "dynamic medium action shot of Minh Khang turning sharply toward a misty forest path at the edge of town, "
                "dark navy trench coat swirling, pointing urgently toward a towering vertical crimson energy glow "
                "rising from the distant northern woods, Linh Chi beside him ready to run"
            ),
            motion_prompt=(
                "dynamic handheld push-in camera, Minh Khang's coat billowing in wind, glowing red embers drifting "
                "through the dark mist, urgent expressive gesture"
            ),
            sfx_cue="wind_ominous",
            lead_in_sec=0.25,
            tail_pad_sec=0.50,
            keyframe_path="assets/keyframes/shot_005.png",
        ),
        Shot(
            shot_id="shot_006",
            character="narrator",
            character_lora=None,
            voice_ref="edge_tts:vi-VN-NamMinhNeural",
            voice_spec=characters["narrator"],
            shot_type=ShotType.KEN_BURNS_TILT_UP,
            dialogue="Và khi hai người bước vào màn sương đỏ, cánh cổng bị phong ấn suốt nghìn năm bắt đầu thức tỉnh.",
            image_prompt=(
                "epic wide rear-view silhouette shot of Minh Khang and Linh Chi standing side by side at the entrance "
                "of a mist-shrouded ancient northern forest, looking up at a massive weathered stone torii-like ruin "
                "archway cracked with glowing crimson runes and swirling red mist awakening inside the gate"
            ),
            motion_prompt=None,
            sfx_cue="portal_hum",
            lead_in_sec=0.35,
            tail_pad_sec=1.20,
            keyframe_path="assets/keyframes/shot_006.png",
        ),
    ]

    return EpisodeScript(
        episode_id="ep_01_pilot",
        title="Đêm Bầu Trời Đỏ — Tập Thử Nghiệm MVP",
        fps=24,
        width=1280,
        height=720,
        style_lora="anime_cinematic_v1.safetensors",
        characters=characters,
        shots=shots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate script.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "script.json",
        help="Destination path for script.json",
    )
    args = parser.parse_args()

    episode = build_pilot_episode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(episode.model_dump_json(indent=2), encoding="utf-8")

    ai_n, kb_n, ai_ratio = episode.dynamic_ratio()
    print(f"Validated script written to {args.output}")
    print(f"Shots: {len(episode.shots)} total | video_ai={ai_n} ({ai_ratio:.0%}) | ken_burns={kb_n} ({1-ai_ratio:.0%})")


if __name__ == "__main__":
    main()
