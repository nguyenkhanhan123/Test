import gc
from pathlib import Path
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


CLIPS = {
    "battle_brawl_city.mp4": "3D CGI Unreal Engine 5 tokusatsu super sentai rangers in red blue green PBR metallic armor fighting dark biomechanical monsters in ruined city street, dynamic martial arts kicks, punches, raytraced sparks flying, fast camera tracking, cinematic 3D movie",
    "battle_villain_ambush.mp4": "3D CGI Unreal Engine 5 menacing dark armored alien monster unleashing purple volumetric shockwave energy blast in city ruins, debris flying, dramatic 3D tokusatsu battle",
    "battle_wind_slash.mp4": "3D CGI Unreal Engine 5 green armored sentai warrior leaping in mid-air performing high speed spinning kick and slashing with emerald energy blade tornado, raytraced sparks exploding",
    "battle_water_blast.mp4": "3D CGI Unreal Engine 5 blue armored sentai warrior firing high pressure swirling blue water energy beam at monster, dynamic dodge flip, photorealistic 3D action",
    "battle_fire_punch.mp4": "3D CGI Unreal Engine 5 red armored sentai warrior charging blazing fire fist and delivering devastating uppercut punch to dark monster, huge fiery explosion shockwave",
    "henshin_fire.mp4": "3D CGI Unreal Engine 5 red ranger warrior transforming, PBR crimson metallic armor pieces flying and locking onto body, blazing fire aura sparks, dynamic camera orbit",
    "henshin_water.mp4": "3D CGI Unreal Engine 5 blue ranger warrior transforming, cobalt metallic armor assembling with swirling ocean water vortex, glowing visor activation",
    "henshin_wind.mp4": "3D CGI Unreal Engine 5 green ranger warrior transforming, emerald aerodynamic armor locking in with green wind cyclone, dramatic volumetric lighting",
    "rollcall_explosion.mp4": "3D CGI Unreal Engine 5 five colorful tokusatsu super sentai rangers posing heroically in formation, massive colorful smoke explosions erupting behind them in red blue yellow green pink",
    "gokai_slash_finisher.mp4": "3D CGI Unreal Engine 5 red sentai warrior charging glowing elemental energy sword and unleashing a massive fiery crescent slash wave destroying dark monster in huge explosion",
    "mecha_combine.mp4": "3D CGI Unreal Engine 5 five giant mechanical elemental beast robots combining in mid-air into one colossal heroic armored mecha titan, glowing golden core, dramatic sky",
    "mecha_sword_finish.mp4": "3D CGI Unreal Engine 5 giant heroic armored mecha robot swinging a colossal glowing energy greatsword, huge explosion in background sky, epic 3D movie",
}


def main():
    if not torch.cuda.is_available():
        print("⚠️ Không có GPU CUDA — bỏ qua bước tạo Stock Clips.")
        return

    stock_dir = Path("assets/stock_clips")
    stock_dir.mkdir(parents=True, exist_ok=True)

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    for fname, prompt in CLIPS.items():
        out_path = stock_dir / fname
        if out_path.exists():
            print("Skip (đã có):", fname)
            continue
        print("Generating LTX-Video Clip (512x320, 41f):", fname)
        frames = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, static, watermark, text",
            width=512,
            height=320,
            num_frames=41,
            num_inference_steps=24,
            guidance_scale=3.0,
        ).frames[0]
        export_to_video(frames, str(out_path), fps=24)
        print("Saved:", out_path)
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
