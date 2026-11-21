import gc
from pathlib import Path
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video


CLIPS = {
    "battle_brawl_city.mp4": "3D CGI Unreal Engine 5 tokusatsu super sentai rangers in red blue green PBR metallic armor fighting dark biomechanical monsters in ruined city street, dynamic martial arts kicks, punches, raytraced sparks flying, cinematic 3D movie, masterpiece",
    "battle_villain_ambush.mp4": "3D CGI Unreal Engine 5 menacing dark armored alien monster unleashing purple volumetric shockwave energy blast in city ruins, debris flying, dramatic 3D tokusatsu battle, masterpiece",
    "battle_wind_slash.mp4": "3D CGI Unreal Engine 5 green armored sentai warrior leaping in mid-air performing high speed spinning kick and slashing with emerald energy blade tornado, raytraced sparks exploding, masterpiece",
    "battle_water_blast.mp4": "3D CGI Unreal Engine 5 blue armored sentai warrior firing high pressure swirling blue water energy beam at monster, dynamic dodge flip, photorealistic 3D action, masterpiece",
    "battle_fire_punch.mp4": "3D CGI Unreal Engine 5 red armored sentai warrior charging blazing fire fist and delivering devastating uppercut punch to dark monster, huge fiery explosion shockwave, masterpiece",
    "henshin_fire.mp4": "3D CGI Unreal Engine 5 red ranger warrior transforming, PBR crimson metallic armor pieces flying and locking onto body, blazing fire aura sparks, dynamic camera orbit, masterpiece",
    "henshin_water.mp4": "3D CGI Unreal Engine 5 blue ranger warrior transforming, cobalt metallic armor assembling with swirling ocean water vortex, glowing visor activation, masterpiece",
    "henshin_wind.mp4": "3D CGI Unreal Engine 5 green ranger warrior transforming, emerald aerodynamic armor locking in with green wind cyclone, dramatic volumetric lighting, masterpiece",
    "rollcall_explosion.mp4": "3D CGI Unreal Engine 5 five colorful tokusatsu super sentai rangers posing heroically in formation, massive colorful smoke explosions erupting behind them in red blue yellow green pink, masterpiece",
    "gokai_slash_finisher.mp4": "3D CGI Unreal Engine 5 red sentai warrior charging glowing elemental energy sword and unleashing a massive fiery crescent slash wave destroying dark monster in huge explosion, masterpiece",
    "mecha_combine.mp4": "3D CGI Unreal Engine 5 five giant mechanical elemental beast robots combining in mid-air into one colossal heroic armored mecha titan, glowing golden core, dramatic sky, masterpiece",
    "mecha_sword_finish.mp4": "3D CGI Unreal Engine 5 giant heroic armored mecha robot swinging a colossal glowing energy greatsword, huge explosion in background sky, epic 3D movie, masterpiece",
}


def main():
    import shutil
    locks_dir = Path.home() / ".cache" / "huggingface" / "hub" / ".locks"
    if locks_dir.exists():
        shutil.rmtree(locks_dir, ignore_errors=True)

    if not torch.cuda.is_available():
        print("⚠️ Không có GPU CUDA — bỏ qua bước tạo Stock Clips.", flush=True)
        return

    stock_dir = Path("assets/stock_clips")
    stock_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 Đang tải MotionAdapter (AnimateDiff)...", flush=True)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    print("2/3 Đang nạp DreamShaper-8 (3D CGI)...", flush=True)
    pipe = AnimateDiffPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", clip_sample=False
    )
    print("3/3 Đang chuyển model lên GPU T4...", flush=True)
    pipe = pipe.to("cuda")
    pipe.vae.enable_slicing()

    for fname, prompt in CLIPS.items():
        out_path = stock_dir / fname
        if out_path.exists():
            print("Skip (đã có):", fname)
            continue
        print("Generating 3D CGI Clip (576x384, 24f):", fname)
        frames = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, static, watermark, text",
            width=576,
            height=384,
            num_frames=24,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).frames[0]
        export_to_video(frames, str(out_path), fps=12)
        print("Saved:", out_path)

    del pipe, adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
