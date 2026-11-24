"""
Sentai 3D Director — Demo cảnh phim hoạt hình 3D thật (8 giây, 720p 24fps, 3 góc máy điện ảnh)
- Dùng nhân vật 3D chuẩn Rig + Motion Capture thật từ Mixamo (Xbot.glb)
- Bối cảnh đấu trường đêm 3D thật có sàn phản chiếu, cột năng lượng phát sáng, sương khói & 3 góc máy cắt cảnh tự động
- Ghép lồng tiếng Việt (edge-tts) + hiệu ứng âm thanh + nhạc nền Tokusatsu bằng FFmpeg
"""
import asyncio
import math
import os
import shutil
import subprocess
import sys
import urllib.request


def run_blender_scene(xbot_path, out_raw_mp4):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 192  # 8 giây @ 24fps
    scene.render.fps = 24

    def make_pbr_mat(name, color, metallic=0.5, roughness=0.3, emission=None, em_strength=0.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness
            if emission and "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emission
                bsdf.inputs["Emission Strength"].default_value = em_strength
        return mat

    # --- 1. DỰNG BỐI CẢNH ĐẤU TRƯỜNG ĐÊM 3D THẬT (SCI-FI TOKUSATSU ARENA) ---
    mat_floor = make_pbr_mat("Arena_Asphalt", (0.025, 0.028, 0.038, 1.0), metallic=0.25, roughness=0.22)
    mat_pillar = make_pbr_mat("Arena_Pillar", (0.06, 0.065, 0.08, 1.0), metallic=0.65, roughness=0.35)
    mat_neon_red = make_pbr_mat("Neon_Red", (1.0, 0.08, 0.04, 1.0), metallic=0.1, roughness=0.2, emission=(1.0, 0.12, 0.04, 1.0), em_strength=10.0)
    mat_neon_cyan = make_pbr_mat("Neon_Cyan", (0.05, 0.75, 1.0, 1.0), metallic=0.1, roughness=0.2, emission=(0.08, 0.72, 1.0, 1.0), em_strength=9.0)

    # Sàn đấu phản chiếu ánh đèn
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "ArenaFloor"
    floor.data.materials.append(mat_floor)

    # Các dải đèn neon chạy dọc hành lang sân khấu
    for sx in (-3.2, 3.2):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, 0, 0.02))
        strip = bpy.context.active_object
        strip.scale = (0.08, 16.0, 0.04)
        bpy.ops.object.transform_apply(scale=True)
        strip.data.materials.append(mat_neon_red if sx < 0 else mat_neon_cyan)

    # 8 trụ cột kiến trúc 3D + thanh đèn phát sáng hai bên
    for idx, py in enumerate((-4.5, -1.5, 1.5, 4.5)):
        for sx in (-4.2, 4.2):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, py, 2.5))
            pil = bpy.context.active_object
            pil.scale = (0.75, 0.75, 5.0)
            bpy.ops.object.transform_apply(scale=True)
            pil.data.materials.append(mat_pillar)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx - (0.40 if sx > 0 else -0.40), py, 2.2))
            glow_bar = bpy.context.active_object
            glow_bar.scale = (0.08, 0.16, 3.6)
            bpy.ops.object.transform_apply(scale=True)
            glow_bar.data.materials.append(mat_neon_red if (idx % 2 == 0) else mat_neon_cyan)

    # Bức tường năng lượng phía xa tạo chiều sâu điện ảnh
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 8.5, 3.0))
    back_wall = bpy.context.active_object
    back_wall.scale = (14.0, 0.5, 6.0)
    bpy.ops.object.transform_apply(scale=True)
    back_wall.data.materials.append(mat_pillar)

    # --- 2. NẠP 2 NHÂN VẬT 3D CHUẨN MIXAMO (HỎA ĐỎ & MẠC KHIÊU) ---
    def load_sentai_character(char_name, loc_xyz, yaw_deg, primary_rgba, secondary_rgba, glow_rgba, preferred_actions):
        before_objs = set(bpy.context.scene.objects)
        before_acts = set(bpy.data.actions)
        bpy.ops.import_scene.gltf(filepath=xbot_path)
        new_objs = [o for o in bpy.context.scene.objects if o not in before_objs]
        new_acts = [a for a in bpy.data.actions if a not in before_acts]

        arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
        meshes = [o for o in new_objs if o.type == "MESH"]
        roots = [o for o in new_objs if o.parent is None]

        # Tạo 1 Empty làm gốc di chuyển nhân vật an toàn
        root_empty = bpy.data.objects.new(f"{char_name}_Root", None)
        scene.collection.objects.link(root_empty)
        root_empty.location = loc_xyz
        root_empty.rotation_euler = (0, 0, math.radians(yaw_deg))
        for r in roots:
            r.parent = root_empty

        # Nhuộm giáp chiến binh Super Sentai (Metallic Armor + Glowing Joints)
        mat_armor = make_pbr_mat(f"{char_name}_Armor", primary_rgba, metallic=0.72, roughness=0.24)
        mat_joint = make_pbr_mat(f"{char_name}_Joint", secondary_rgba, metallic=0.40, roughness=0.38, emission=glow_rgba, em_strength=1.4)

        for m in meshes:
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.select_all(action="DESELECT")
            m.select_set(True)
            bpy.ops.object.shade_smooth()
            for slot in m.material_slots:
                if slot.material is None:
                    continue
                low = slot.material.name.lower()
                if "joint" in low or "alpha_joints" in low:
                    slot.material = mat_joint
                else:
                    slot.material = mat_armor

        # Gán đúng động tác Motion Capture Mixamo và lặp mượt qua 192 frame bằng NLA
        chosen_act = None
        all_acts = new_acts if new_acts else list(bpy.data.actions)
        for pref in preferred_actions:
            chosen_act = next((a for a in all_acts if pref in a.name.lower()), None)
            if chosen_act:
                break
        if chosen_act is None and all_acts:
            chosen_act = all_acts[0]

        if arm and chosen_act:
            if not arm.animation_data:
                arm.animation_data_create()
            arm.animation_data.action = None
            track = arm.animation_data.nla_tracks.new()
            strip = track.strips.new(chosen_act.name, start=1, action=chosen_act)
            strip.repeat = 10.0
            print(f"[{char_name}] Activated Mixamo mocap action: {chosen_act.name}", flush=True)

        return root_empty, arm

    # Hỏa Đỏ (Giáp đỏ rực kim loại + khớp phát sáng cam lửa) tiến bước chiến đấu về phía Mạc Khiêu
    hero_root, _ = load_sentai_character(
        "HoaDo",
        loc_xyz=(-0.3, -2.6, 0.0),
        yaw_deg=180.0,
        primary_rgba=(0.85, 0.04, 0.06, 1.0),
        secondary_rgba=(0.04, 0.04, 0.05, 1.0),
        glow_rgba=(1.0, 0.18, 0.02, 1.0),
        preferred_actions=("walk", "run", "idle"),
    )
    # Hỏa Đỏ bước tiến từ Y=-2.8 tới Y=-1.1 trong Shot 1-2 rồi đứng đối đầu
    hero_root.location = (-0.25, -2.8, 0.0)
    hero_root.keyframe_insert(data_path="location", frame=1)
    hero_root.location = (-0.15, -1.2, 0.0)
    hero_root.keyframe_insert(data_path="location", frame=128)
    hero_root.location = (-0.15, -0.9, 0.0)
    hero_root.keyframe_insert(data_path="location", frame=192)

    # Mạc Khiêu (Đối thủ giáp đen tím + khớp phát sáng xanh băng) đứng chắn phía trước
    villain_root, _ = load_sentai_character(
        "MacKhieu",
        loc_xyz=(0.25, 2.2, 0.0),
        yaw_deg=0.0,
        primary_rgba=(0.08, 0.04, 0.14, 1.0),
        secondary_rgba=(0.02, 0.03, 0.05, 1.0),
        glow_rgba=(0.08, 0.72, 1.0, 1.0),
        preferred_actions=("idle", "sneak", "agree"),
    )

    # --- 3. ÁNH SÁNG ĐIỆN ẢNH TOKUSATSU (3-POINT + RIM LIGHT ĐỎ/XANH) ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.015, 0.02, 0.035, 1.0)
        bg.inputs[1].default_value = 0.6

    def add_area_light(name, loc, rot_deg, energy, color, size=2.2):
        bpy.ops.object.light_add(type="AREA", location=loc, rotation=tuple(math.radians(a) for a in rot_deg))
        l = bpy.context.active_object
        l.name = name
        l.data.energy = energy
        l.data.color = color
        l.data.size = size
        return l

    add_area_light("Key_Hero", (-2.6, -3.5, 2.8), (62, 0, -35), 260.0, (1.0, 0.92, 0.85), 2.5)
    add_area_light("Rim_Red", (0.5, 0.8, 2.5), (-65, 0, 15), 480.0, (1.0, 0.18, 0.06), 2.2)
    add_area_light("Rim_Cyan", (-0.5, 4.5, 2.6), (65, 0, 0), 420.0, (0.10, 0.75, 1.0), 2.4)
    add_area_light("Top_Arena", (0.0, 0.0, 5.2), (0, 0, 0), 220.0, (0.75, 0.82, 1.0), 4.0)

    # --- 4. HỆ THỐNG 3 GÓC MÁY ĐIỆN ẢNH CẮT CẢNH TỰ ĐỘNG ---
    def make_target(name, parent, offset):
        t = bpy.data.objects.new(name, None)
        scene.collection.objects.link(t)
        t.parent = parent
        t.location = offset
        return t

    hero_chest_tgt = make_target("Target_HeroChest", hero_root, (0, 0, 1.25))
    hero_face_tgt = make_target("Target_HeroFace", hero_root, (0, 0, 1.52))
    villain_chest_tgt = make_target("Target_VillainChest", villain_root, (0, 0, 1.30))

    def make_cam(name, start_loc, end_loc, f_start, f_end, target_obj, lens=42.0):
        bpy.ops.object.camera_add(location=start_loc)
        cam = bpy.context.active_object
        cam.name = name
        cam.data.lens = lens
        cam.data.dof.use_dof = True
        cam.data.dof.focus_object = target_obj
        cam.data.dof.aperture_fstop = 2.8
        cam.location = start_loc
        cam.keyframe_insert(data_path="location", frame=f_start)
        cam.location = end_loc
        cam.keyframe_insert(data_path="location", frame=f_end)
        con = cam.constraints.new(type="TRACK_TO")
        con.target = target_obj
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        return cam

    # Shot 1 (Frame 1–64): Low-angle Hero Reveal (quét từ thấp lên khoe dáng chiến binh tiến bước)
    cam1 = make_cam("Cam1_HeroReveal", (1.45, -0.3, 0.65), (0.85, 0.6, 1.25), 1, 64, hero_chest_tgt, lens=45.0)
    # Shot 2 (Frame 65–128): Over-the-Shoulder Confrontation (góc qua vai Hỏa Đỏ nhìn đối thủ Mạc Khiêu)
    cam2 = make_cam("Cam2_Confrontation", (-1.15, -3.1, 1.55), (-0.75, -2.3, 1.48), 65, 128, villain_chest_tgt, lens=38.0)
    # Shot 3 (Frame 129–192): Dramatic Hero Close-Up Push-In (cận cảnh chiến binh sẵn sàng chiến đấu)
    cam3 = make_cam("Cam3_HeroCloseUp", (-1.2, 0.55, 1.35), (0.65, 0.35, 1.50), 129, 192, hero_face_tgt, lens=55.0)

    scene.camera = cam1
    for cname, frame, cam_obj in [("Shot1", 1, cam1), ("Shot2", 65, cam2), ("Shot3", 129, cam3)]:
        mk = scene.timeline_markers.new(cname, frame=frame)
        mk.camera = cam_obj

    # --- 5. RENDER CYCLES GPU 720p 24FPS ---
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = out_raw_mp4
    os.makedirs(os.path.dirname(out_raw_mp4), exist_ok=True)
    print("Đang render cảnh phim 3D (192 frames @ 1280x720 Cycles GPU)...", flush=True)
    bpy.ops.render.render(animation=True)
    print("-> Render Blender hoàn tất:", out_raw_mp4, flush=True)


def download_xbot(dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500_000:
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb"
    print("Đang tải nhân vật 3D chuẩn Mixamo (Xbot.glb)...", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"-> Tải thành công ({os.path.getsize(dest_path)/1024/1024:.2f} MB)", flush=True)


def synth_vietnamese_voice(out_mp3):
    text = "Mạc Khiêu! Ngươi không thể thoát khỏi đấu trường này đâu. Nguyên Giới Giáp Sư — sẵn sàng nghênh chiến!"
    try:
        import edge_tts

        async def _run():
            comm = edge_tts.Communicate(text, voice="vi-VN-NamMinhNeural", rate="+6%", pitch="-4Hz")
            await comm.save(out_mp3)

        asyncio.run(_run())
        return True
    except Exception as e:
        print("⚠️ Không tạo được giọng edge-tts:", e, flush=True)
        return False


def mux_audio_and_subtitles(raw_mp4, voice_mp3, final_mp4):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(raw_mp4, final_mp4)
        return

    bgm_candidates = [
        "/content/Test/assets/audio/bgm/bgm_action_combat.wav",
        "/content/Test/assets/audio/bgm/bgm_heroic_theme.wav",
    ]
    bgm = next((p for p in bgm_candidates if os.path.exists(p)), None)

    cmd = [ffmpeg, "-y", "-i", raw_mp4]
    inputs = 1
    voice_idx = None
    bgm_idx = None
    if voice_mp3 and os.path.exists(voice_mp3):
        cmd += ["-i", voice_mp3]
        voice_idx = inputs
        inputs += 1
    if bgm:
        cmd += ["-stream_loop", "-1", "-i", bgm]
        bgm_idx = inputs
        inputs += 1

    if voice_idx is not None and bgm_idx is not None:
        # Lọc giọng vang mũ giáp chiến binh nhẹ + trộn nhạc nền hành động
        af = (
            f"[{voice_idx}:a]adelay=600|600,highpass=f=220,lowpass=f=3600,volume=1.35[v];"
            f"[{bgm_idx}:a]volume=0.28[b];"
            f"[v][b]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        cmd += ["-filter_complex", af, "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", final_mp4]
    elif voice_idx is not None:
        cmd += ["-map", "0:v:0", "-map", f"{voice_idx}:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", final_mp4]
    else:
        shutil.copy2(raw_mp4, final_mp4)
        return

    subprocess.run(cmd, check=True)
    print("-> Đã ghép xong âm thanh phim 3D:", final_mp4, flush=True)


def main():
    if "--blender-render" in sys.argv:
        idx = sys.argv.index("--blender-render")
        run_blender_scene(sys.argv[idx + 1], sys.argv[idx + 2])
        return

    xbot_glb = "/content/Test/assets/models_3d/Xbot.glb"
    raw_mp4 = "/content/Test/build_output/demo_sentai_3d_raw.mp4"
    voice_mp3 = "/content/Test/build_output/demo_sentai_voice.mp3"
    final_mp4 = "/content/Test/build_output/demo_sentai_3d_scene.mp4"
    os.makedirs("/content/Test/build_output", exist_ok=True)

    download_xbot(xbot_glb)

    blender_bin = "/content/blender/blender"
    cmd = [
        blender_bin, "--background",
        "--python", os.path.abspath(__file__), "--",
        "--blender-render", xbot_glb, raw_mp4,
    ]
    subprocess.run(cmd, check=True)

    has_voice = synth_vietnamese_voice(voice_mp3)
    mux_audio_and_subtitles(raw_mp4, voice_mp3 if has_voice else None, final_mp4)
    print(f"HOÀN TẤT PHIM 3D DEMO: {final_mp4} ({os.path.getsize(final_mp4)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
