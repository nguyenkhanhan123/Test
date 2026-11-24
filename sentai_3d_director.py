"""
Sentai 3D Director — Cảnh Võ Thuật NGƯỜI THẬT 1 Phút (60 giây, 720p 24fps)
Dùng 100% NHÂN VẬT NGƯỜI THẬT + MOTION CAPTURE VÕ THUẬT THẬT TỪ MIXAMO (tự tải từ Hugging Face):
- Dataset: Linzhan/Mixamo-Animations-Characters (114 nhân vật chuẩn Rig + 2.317 file Motion Capture .fbx thật!)
- Không tự viết tay góc xoay xương -> chân bám chặt mặt đất, đấm/đá xoay/né đòn/trúng đòn thật 100% như diễn viên đóng thế!
- Tự động nối mượt các clip .fbx bằng Blender NLA (blend_in=6 frame) suốt 60 giây + 8 góc máy hành động + SFX & thoại tiếng Việt!
"""
import asyncio
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import wave


def download_mixamo_assets(work_dir):
    """Tự động tải 2 nhân vật người thật + bộ Motion Capture võ thuật thật (.fbx) từ Hugging Face Mixamo dataset."""
    from huggingface_hub import HfApi, hf_hub_download

    manifest_path = os.path.join(work_dir, "mixamo_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if os.path.exists(data.get("hero_fbx", "")) and os.path.exists(data.get("enemy_fbx", "")) and len(data.get("anims", {})) >= 5:
            print("Dùng lại bộ nhân vật & Motion Capture Mixamo đã tải sẵn!", flush=True)
            return data

    api = HfApi()
    repo_id = None
    files = []
    for cand in ("Linzhan/Mixamo-Animations-Characters", "tanish434/Mixamo-Animations-Characters"):
        try:
            print(f"Đang quét kho Mixamo Motion Capture trên Hugging Face ({cand})...", flush=True)
            files = api.list_repo_files(repo_id=cand, repo_type="dataset")
            if files:
                repo_id = cand
                break
        except Exception as e:
            print(f"  Bỏ qua {cand}: {e}", flush=True)

    if not repo_id or not files:
        raise RuntimeError("Không truy cập được kho Mixamo trên Hugging Face.")

    # 1. Chọn 2 nhân vật NGƯỜI THẬT NAM từ character_refined/ (hoặc character/)
    char_fbxs = [f for f in files if f.lower().endswith(".fbx") and ("character_refined/" in f.lower() or f.lower().startswith("character/"))]
    if not char_fbxs:
        char_fbxs = [f for f in files if f.lower().endswith(".fbx") and "character" in f.lower()]

    def pick_char(preferred_names, exclude=None):
        for name in preferred_names:
            for f in char_fbxs:
                if f == exclude:
                    continue
                base = os.path.basename(f).lower()
                if name.lower() in base:
                    return f
        # Fallback: lấy file fbx đầu tiên khác exclude
        for f in char_fbxs:
            if f != exclude:
                return f
        return None

    hero_remote = pick_char(["remy", "david", "douglas", "leonard", "adam", "brian", "liam", "malcolm", "james", "bryce", "ch31", "ch17", "ch14"])
    enemy_remote = pick_char(["swat", "soldier", "malcolm", "douglas", "leonard", "brian", "david", "josh", "pete", "ch15", "ch33", "vanguard"], exclude=hero_remote)
    if not hero_remote or not enemy_remote:
        raise RuntimeError(f"Không tìm thấy đủ 2 file nhân vật FBX trong {repo_id}.")

    # 2. Chọn bộ file MOTION CAPTURE VÕ THUẬT THẬT (.fbx)
    anim_fbxs = [f for f in files if f.lower().endswith(".fbx") and "animation" in f.lower()]
    # Ưu tiên thư mục animation_motion_ybot hoặc animation/ chuẩn nhất
    ybot_anims = [f for f in anim_fbxs if "ybot" in f.lower()]
    if len(ybot_anims) > 100:
        anim_fbxs = ybot_anims

    def pick_anim(keywords, used_set):
        for kw in keywords:
            k = kw.lower()
            for f in anim_fbxs:
                if f in used_set:
                    continue
                base = os.path.basename(f).lower().replace("_", " ").replace("-", " ")
                if k in base:
                    used_set.add(f)
                    return f
        # Nếu trùng used_set thì cho phép dùng lại
        for kw in keywords:
            k = kw.lower()
            for f in anim_fbxs:
                base = os.path.basename(f).lower().replace("_", " ").replace("-", " ")
                if k in base:
                    return f
        return None

    used = set()
    anim_queries = {
        "idle": ["fight idle", "fighting idle", "bouncing fight", "boxing idle", "combat idle", "idle"],
        "jab": ["lead jab", "jab", "punching", "boxing", "punch"],
        "cross": ["cross punch", "hook punch", "right hook", "punch combo", "hook", "punch"],
        "uppercut": ["uppercut", "elbow punch", "punch to elbow", "quad punch", "boxing"],
        "kick_front": ["mma kick", "front kick", "push kick", "kicking", "kick"],
        "kick_spin": ["roundhouse kick", "hurricane kick", "flying kick", "spinning", "martelo", "kick"],
        "block": ["center block", "body block", "block", "dodging", "dodge"],
        "hit": ["hit to body", "head hit", "stomach hit", "kidney hit", "reaction", "hit"],
        "knockdown": ["knocked down", "falling back", "fall flat", "dying", "defeat", "hit"],
    }

    selected_anims_remote = {}
    for slot, kws in anim_queries.items():
        found = pick_anim(kws, used)
        if found:
            selected_anims_remote[slot] = found

    print(f"Nhân vật Hero:  {hero_remote}", flush=True)
    print(f"Nhân vật Enemy: {enemy_remote}", flush=True)
    print(f"Đã chọn {len(selected_anims_remote)} động tác Motion Capture võ thuật thật:", flush=True)
    for k, v in selected_anims_remote.items():
        print(f"  - {k:10s}: {os.path.basename(v)}", flush=True)

    # Tải các file đã chọn về ổ đĩa Colab
    local_dir = os.path.join(work_dir, "mixamo_cache")
    os.makedirs(local_dir, exist_ok=True)

    def fetch(remote_path):
        return hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="dataset", local_dir=local_dir)

    hero_fbx = fetch(hero_remote)
    enemy_fbx = fetch(enemy_remote)
    anims_local = {k: fetch(v) for k, v in selected_anims_remote.items()}

    manifest = {"hero_fbx": hero_fbx, "enemy_fbx": enemy_fbx, "anims": anims_local}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_blender_mocap_fight(manifest_path, out_raw_mp4):
    import bpy
    from mathutils import Vector

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    FPS = 24
    TOTAL_FRAMES = 60 * FPS  # 1440 frames = 60 giây chuẩn
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.fps = FPS

    def make_pbr_mat(name, color, metallic=0.2, roughness=0.4, emission=None, em_strength=0.0):
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

    # --- 1. BỐI CẢNH ĐẤU TRƯỜNG SÂN THƯỢNG ĐÊM 3D ---
    mat_floor = make_pbr_mat("Rooftop_Floor", (0.03, 0.035, 0.045, 1.0), metallic=0.18, roughness=0.26)
    mat_concrete = make_pbr_mat("Concrete_Wall", (0.07, 0.075, 0.09, 1.0), metallic=0.12, roughness=0.65)
    mat_neon_red = make_pbr_mat("Neon_Red", (1.0, 0.10, 0.05, 1.0), metallic=0.1, roughness=0.2, emission=(1.0, 0.14, 0.05, 1.0), em_strength=10.0)
    mat_neon_blue = make_pbr_mat("Neon_Blue", (0.08, 0.65, 1.0, 1.0), metallic=0.1, roughness=0.2, emission=(0.08, 0.68, 1.0, 1.0), em_strength=9.0)
    mat_spark = make_pbr_mat("Impact_Spark", (1.0, 0.45, 0.08, 1.0), metallic=0.0, roughness=0.1, emission=(1.0, 0.35, 0.05, 1.0), em_strength=28.0)

    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0, 0, 0))
    bpy.context.active_object.data.materials.append(mat_floor)

    for sx in (-3.6, 3.6):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, 0, 0.02))
        st = bpy.context.active_object
        st.scale = (0.06, 16.0, 0.04)
        bpy.ops.object.transform_apply(scale=True)
        st.data.materials.append(mat_neon_red if sx < 0 else mat_neon_blue)

    for idx, py in enumerate((-5.0, -2.0, 1.0, 4.0)):
        for sx in (-4.8, 4.8):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, py, 2.4))
            p = bpy.context.active_object
            p.scale = (0.65, 0.65, 4.8)
            bpy.ops.object.transform_apply(scale=True)
            p.data.materials.append(mat_concrete)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx - (0.35 if sx > 0 else -0.35), py, 2.2))
            gb = bpy.context.active_object
            gb.scale = (0.07, 0.14, 3.2)
            bpy.ops.object.transform_apply(scale=True)
            gb.data.materials.append(mat_neon_red if idx % 2 == 0 else mat_neon_blue)

    for bx, by, bh in [(-7, 10, 8), (-3, 12, 11), (2, 11, 9), (6.5, 10, 12)]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bh / 2.0))
        b = bpy.context.active_object
        b.scale = (2.6, 2.2, bh)
        bpy.ops.object.transform_apply(scale=True)
        b.data.materials.append(mat_concrete)

    # --- 2. NẠP BỘ MOTION CAPTURE ACTIONS (.FBX) & CHUẨN HÓA ĐỂ KHÔNG TRÔI VỊ TRÍ X/Y ---
    mocap_actions = {}
    for clip_key, fbx_path in manifest["anims"].items():
        before_objs = set(bpy.context.scene.objects)
        before_acts = set(bpy.data.actions)
        try:
            bpy.ops.import_scene.fbx(filepath=fbx_path, ignore_leaf_bones=True)
        except Exception as e:
            print(f"⚠️ Không nạp được anim {clip_key}: {e}", flush=True)
            continue
        new_objs = [o for o in bpy.context.scene.objects if o not in before_objs]
        new_acts = [a for a in bpy.data.actions if a not in before_acts]

        # Xóa bộ xương tạm của file anim, chỉ giữ lại Action Motion Capture!
        for o in new_objs:
            bpy.data.objects.remove(o, do_unlink=True)

        if not new_acts:
            continue
        act = new_acts[0]
        act.name = f"Mocap_{clip_key}"
        act.use_fake_user = True

        # Khóa trượt ngang (X, Y) của xương Hips tại frame đầu để nhân vật KHÔNG trôi ra khỏi sân đấu,
        # nhưng GIỮ NGUYÊN 100% độ cao Z & mọi chuyển động chân/eo/tay/đầu để chân bám chặt mặt đất!
        for fc in act.fcurves:
            if "hips" in fc.data_path.lower() and fc.data_path.endswith("location") and fc.array_index in (0, 2):
                # Trong Mixamo FBX, index 0 = X ngang, index 2 = Z tiến/lùi (index 1 = Y độ cao thẳng đứng!)
                if fc.keyframe_points:
                    base_v = fc.keyframe_points[0].co.y
                    for kp in fc.keyframe_points:
                        # Giữ lại 25% nhịp bước tự nhiên để đòn đánh có lực nhưng không trôi mất
                        kp.co.y = base_v + (kp.co.y - base_v) * 0.25
                        kp.handle_left.y = base_v + (kp.handle_left.y - base_v) * 0.25
                        kp.handle_right.y = base_v + (kp.handle_right.y - base_v) * 0.25

        f_len = max(12, int(round(act.frame_range[1] - act.frame_range[0])))
        mocap_actions[clip_key] = (act, f_len)
        print(f"Loaded REAL Mocap Action: {act.name} ({f_len} frames)", flush=True)

    if not mocap_actions:
        raise RuntimeError("Không nạp được Action Motion Capture nào từ Mixamo FBX!")

    # --- 3. NẠP 2 NHÂN VẬT NGƯỜI THẬT MIXAMO (.FBX) ---
    def load_mixamo_character(name, fbx_path):
        before_objs = set(bpy.context.scene.objects)
        bpy.ops.import_scene.fbx(filepath=fbx_path, ignore_leaf_bones=True)
        new_objs = [o for o in bpy.context.scene.objects if o not in before_objs]
        arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
        meshes = [o for o in new_objs if o.type == "MESH"]
        roots = [o for o in new_objs if o.parent is None]
        if arm is None:
            raise RuntimeError(f"No armature in {fbx_path}")

        if arm.animation_data:
            arm.animation_data.action = None

        # Chuẩn hóa chiều cao đúng 1.78m đứng chạm sàn Z=0
        bpy.context.view_layer.update()
        pts = []
        for m in meshes:
            pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
        min_z = min(v.z for v in pts)
        max_z = max(v.z for v in pts)
        sc = 1.78 / max(0.2, max_z - min_z)

        root_empty = bpy.data.objects.new(f"{name}_Root", None)
        scene.collection.objects.link(root_empty)
        for r in roots:
            r.scale = (r.scale.x * sc, r.scale.y * sc, r.scale.z * sc)
            r.location.z -= min_z * sc
            r.parent = root_empty

        for m in meshes:
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.select_all(action="DESELECT")
            m.select_set(True)
            bpy.ops.object.shade_smooth()

        # Đảm bảo tên xương trùng khớp với Action (nhiều bản Mixamo có/không có tiền tố mixamorig:)
        arm_bone_names = {b.name for b in arm.data.bones}
        sample_act = next(iter(mocap_actions.values()))[0]
        act_has_prefix = any("mixamorig:" in fc.data_path for fc in sample_act.fcurves)
        arm_has_prefix = any("mixamorig:" in bn for bn in arm_bone_names)
        if act_has_prefix and not arm_has_prefix:
            for b in arm.data.bones:
                if not b.name.startswith("mixamorig:"):
                    b.name = "mixamorig:" + b.name
        elif not act_has_prefix and arm_has_prefix:
            for b in arm.data.bones:
                if b.name.startswith("mixamorig:"):
                    b.name = b.name.replace("mixamorig:", "", 1)

        if not arm.animation_data:
            arm.animation_data_create()
        print(f"[{name}] Loaded real human Mixamo character: {len(arm.data.bones)} bones, {len(meshes)} meshes", flush=True)
        return root_empty, arm

    hero_root, hero_arm = load_mixamo_character("HoaDo", manifest["hero_fbx"])
    enemy_root, enemy_arm = load_mixamo_character("MacKhieu", manifest["enemy_fbx"])

    # Đặt 2 nhân vật đối mặt nhau trên trục Y (Mixamo FBX mặc định mặt nhìn về -Y)
    # Hero đứng ở Y=-0.75 quay mặt về +Y (rot Z = 180°); Enemy đứng ở Y=+0.75 quay mặt về -Y (rot Z = 0°)
    hero_root.location = (0.0, -0.78, 0.0)
    hero_root.rotation_euler = (0.0, 0.0, math.radians(180.0))
    enemy_root.location = (0.0, 0.78, 0.0)
    enemy_root.rotation_euler = (0.0, 0.0, 0.0)

    # --- 4. GHÉP CHUỖI MOTION CAPTURE VÕ THUẬT THẬT BẰNG BLENDER NLA (60 GIÂY = 1440 FRAMES) ---
    def schedule_nla(arm, sequence):
        """Nối liên tiếp các clip Motion Capture thật trên NLA với chuyển cảnh mượt (blend_in=6 frame)."""
        track = arm.animation_data.nla_tracks.new()
        track.name = "MartialArts_Mocap"
        cur_frame = 1
        impact_frames = []

        for clip_key, is_attack in sequence:
            if clip_key not in mocap_actions:
                clip_key = next(iter(mocap_actions.keys()))
            act, f_len = mocap_actions[clip_key]
            if cur_frame > TOTAL_FRAMES:
                break
            strip = track.strips.new(f"{clip_key}_{cur_frame}", start=cur_frame, action=act)
            strip.blend_type = "REPLACE"
            strip.extrapolation = "HOLD_FORWARD"
            strip.blend_in = min(6, f_len // 4)
            strip.blend_out = 0
            if is_attack:
                impact_frames.append(cur_frame + int(f_len * 0.45))
            cur_frame += max(18, f_len - 5)

        # Lấp nốt tới frame 1440 bằng idle/knockdown cuối cùng
        if cur_frame <= TOTAL_FRAMES:
            last_key = sequence[-1][0] if sequence[-1][0] in mocap_actions else next(iter(mocap_actions.keys()))
            act, f_len = mocap_actions[last_key]
            strip = track.strips.new(f"hold_{cur_frame}", start=cur_frame, action=act)
            strip.repeat = max(1.0, (TOTAL_FRAMES - cur_frame + 10) / max(1, f_len))
            strip.blend_in = 6
        return impact_frames

    # Biên đạo đối kháng ăn khớp giữa Hỏa Đỏ (tấn công/phản công) và Mạc Khiêu (đỡ/trúng đòn/phản công/gục ngã)
    hero_seq = [
        ("idle", False), ("idle", False),
        ("jab", True), ("cross", True), ("idle", False),
        ("block", False), ("block", False), ("uppercut", True),
        ("kick_front", True), ("idle", False), ("kick_spin", True),
        ("jab", True), ("cross", True), ("block", False),
        ("hit", False), ("block", False), ("uppercut", True),
        ("jab", True), ("cross", True), ("kick_front", True),
        ("kick_spin", True), ("cross", True), ("idle", False),
    ]
    enemy_seq = [
        ("idle", False), ("idle", False),
        ("block", False), ("hit", False), ("idle", False),
        ("jab", True), ("cross", True), ("hit", False),
        ("hit", False), ("idle", False), ("hit", False),
        ("block", False), ("hit", False), ("kick_front", True),
        ("cross", True), ("kick_spin", True), ("hit", False),
        ("hit", False), ("hit", False), ("hit", False),
        ("hit", False), ("knockdown", False), ("knockdown", False),
    ]

    hero_hits = schedule_nla(hero_arm, hero_seq)
    enemy_hits = schedule_nla(enemy_arm, enemy_seq)
    all_impact_frames = sorted(set(f for f in (hero_hits + enemy_hits) if 10 < f < TOTAL_FRAMES - 10))

    # Lưu danh sách frame trúng đòn để tạo tiếng đấm/đá SFX khớp 100%!
    with open(out_raw_mp4 + ".hits.json", "w", encoding="utf-8") as hf:
        json.dump(all_impact_frames, hf)

    # Vòng sóng sáng va chạm (Impact Ring) bung nổ đúng frame trúng đòn
    bpy.ops.mesh.primitive_torus_add(major_radius=0.32, minor_radius=0.025, location=(0, 0, 1.32))
    spark_ring = bpy.context.active_object
    spark_ring.rotation_euler = (math.radians(90), 0, 0)
    spark_ring.data.materials.append(mat_spark)
    spark_ring.scale = (0.001, 0.001, 0.001)
    spark_ring.keyframe_insert(data_path="scale", frame=1)
    for fr in all_impact_frames:
        spark_ring.scale = (0.001, 0.001, 0.001)
        spark_ring.keyframe_insert(data_path="scale", frame=fr - 1)
        spark_ring.scale = (1.5, 1.5, 1.5)
        spark_ring.keyframe_insert(data_path="scale", frame=fr + 2)
        spark_ring.scale = (0.001, 0.001, 0.001)
        spark_ring.keyframe_insert(data_path="scale", frame=fr + 6)

    # --- 5. ÁNH SÁNG ĐIỆN ẢNH 3 ĐIỂM + RIM LIGHT ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.025, 0.04, 1.0)
        bg.inputs[1].default_value = 0.8

    def add_light(name, loc, rot_deg, energy, color, size=2.5):
        bpy.ops.object.light_add(type="AREA", location=loc, rotation=tuple(math.radians(a) for a in rot_deg))
        l = bpy.context.active_object
        l.name = name
        l.data.energy = energy
        l.data.color = color
        l.data.size = size

    add_light("Key_Main", (3.4, -2.4, 3.6), (55, 0, 52), 340.0, (1.0, 0.94, 0.88), 3.0)
    add_light("Fill_Cool", (-3.4, 1.6, 2.8), (64, 0, -115), 190.0, (0.76, 0.86, 1.0), 3.2)
    add_light("Rim_Red", (-2.2, -4.4, 2.5), (68, 0, -25), 400.0, (1.0, 0.22, 0.10), 2.2)
    add_light("Rim_Blue", (2.2, 4.4, 2.5), (-68, 0, 25), 360.0, (0.12, 0.72, 1.0), 2.2)

    # --- 6. 8 GÓC MÁY HÀNH ĐỘNG ĐIỆN ẢNH CẮT CẢNH LIÊN TỤC TRONG 60 GIÂY ---
    mid_tgt = bpy.data.objects.new("FightCenter", None)
    scene.collection.objects.link(mid_tgt)
    mid_tgt.location = (0.0, 0.0, 1.15)

    hero_face_tgt = bpy.data.objects.new("HeroFaceTarget", None)
    scene.collection.objects.link(hero_face_tgt)
    hero_face_tgt.location = (0.0, -0.75, 1.48)

    def make_cam(name, p1, p2, f1, f2, tgt, lens=38.0):
        bpy.ops.object.camera_add(location=p1)
        c = bpy.context.active_object
        c.name = name
        c.data.lens = lens
        c.location = p1
        c.keyframe_insert(data_path="location", frame=f1)
        c.location = p2
        c.keyframe_insert(data_path="location", frame=f2)
        con = c.constraints.new(type="TRACK_TO")
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        return c

    cam_schedule = [
        (1, make_cam("C1_WideIntro", (3.5, -1.2, 1.35), (3.0, 0.4, 1.42), 1, 160, mid_tgt, 35.0)),
        (161, make_cam("C2_HeroClose", (1.1, 0.55, 1.45), (0.85, 0.15, 1.50), 161, 260, hero_face_tgt, 50.0)),
        (261, make_cam("C3_SideActionR", (2.9, -0.2, 1.25), (2.6, 0.5, 1.32), 261, 460, mid_tgt, 38.0)),
        (461, make_cam("C4_LowKickL", (-2.7, -0.5, 0.72), (-2.5, 0.8, 0.98), 461, 680, mid_tgt, 36.0)),
        (681, make_cam("C5_OverShoulder", (-1.5, -2.4, 1.52), (-1.1, -1.8, 1.45), 681, 880, mid_tgt, 40.0)),
        (881, make_cam("C6_CounterR", (2.8, -0.9, 1.15), (2.5, -0.1, 1.32), 881, 1100, mid_tgt, 36.0)),
        (1101, make_cam("C7_FinisherSweep", (-2.4, 0.8, 0.85), (2.2, 1.0, 1.18), 1101, 1320, mid_tgt, 38.0)),
        (1321, make_cam("C8_VictoryHero", (1.5, 0.9, 1.05), (-1.2, 0.8, 1.42), 1321, 1440, hero_face_tgt, 45.0)),
    ]
    scene.camera = cam_schedule[0][1]
    for fstart, cam_obj in cam_schedule:
        mk = scene.timeline_markers.new(cam_obj.name, frame=fstart)
        mk.camera = cam_obj

    # --- 7. RENDER CYCLES GPU SIÊU TỐC (OPTIX RAYTRACING, KHÔNG CPU DENOISE -> ~0.5s/frame) ---
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    gpu_names = []
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type != "CPU"]
                if devs:
                    for d in prefs.devices:
                        d.use = (d.type != "CPU")
                    gpu_names = [f"{d.name} ({dev_type})" for d in devs]
                    break
            except Exception:
                continue
        scene.cycles.device = "GPU" if gpu_names else "CPU"
    except Exception:
        scene.cycles.device = "CPU"

    scene.cycles.use_denoising = False
    scene.cycles.samples = 12
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.08
    print(f"GPU đang dùng: {', '.join(gpu_names) if gpu_names else 'CPU'} | Adaptive 12 samples | Denoise=OFF", flush=True)

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = out_raw_mp4
    os.makedirs(os.path.dirname(out_raw_mp4), exist_ok=True)
    print(f"Đang render 1 phút võ thuật Motion Capture thật ({TOTAL_FRAMES} frames @ 1280x720)...", flush=True)
    bpy.ops.render.render(animation=True)
    print("-> Render Blender hoàn tất:", out_raw_mp4, flush=True)


def generate_impact_sfx_track(out_wav, hit_frames, duration_sec=60.0):
    sr = 44100
    n_samples = int(sr * duration_sec)
    buf = [0.0] * n_samples
    rng = random.Random(2026)
    for fr in hit_frames:
        start = int((fr / 24.0) * sr)
        hit_len = int(0.18 * sr)
        for i in range(hit_len):
            idx = start + i
            if idx >= n_samples:
                break
            t = i / sr
            env = math.exp(-t * 28.0)
            thump = math.sin(2.0 * math.pi * (140.0 * math.exp(-t * 14.0)) * t)
            crack = (rng.random() * 2.0 - 1.0) * math.exp(-t * 45.0)
            buf[idx] = max(-0.95, min(0.95, buf[idx] + (0.75 * thump + 0.45 * crack) * env))

    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(struct.pack("<h", int(s * 32000)) for s in buf))


def synth_dialogue_lines(tmp_dir):
    lines = [
        (2.0, "vi-VN-NamMinhNeural", "+4%", "-2Hz", "Mạc Khiêu! Đứng lại! Hôm nay ta sẽ không để ngươi thoát đâu!"),
        (6.8, "vi-VN-NamMinhNeural", "-6%", "-12Hz", "Chưa biến thân mà dám cản đường ta sao? Nực cười!"),
        (22.5, "vi-VN-NamMinhNeural", "+8%", "-2Hz", "Đừng coi thường võ thuật Nguyên Giới! Xem chiêu này!"),
        (34.5, "vi-VN-NamMinhNeural", "-4%", "-12Hz", "Khá lắm... nhưng tốc độ của ngươi vẫn còn chậm!"),
        (46.0, "vi-VN-NamMinhNeural", "+10%", "0Hz", "Kết thúc rồi! Hỏa Quyền Liên Hoàn!"),
        (55.5, "vi-VN-NamMinhNeural", "+2%", "-2Hz", "Trận chiến này kết thúc tại đây."),
    ]
    out_items = []
    try:
        import edge_tts

        async def _gen():
            for i, (t_sec, voice, rate, pitch, text) in enumerate(lines):
                mp3 = os.path.join(tmp_dir, f"line_{i:02d}.mp3")
                comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
                await comm.save(mp3)
                out_items.append((t_sec, mp3))

        asyncio.run(_gen())
    except Exception as e:
        print("⚠️ Lỗi edge-tts:", e, flush=True)
    return out_items


def mux_full_audio(raw_mp4, sfx_wav, dialogue_items, final_mp4):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(raw_mp4, final_mp4)
        return

    bgm_candidates = [
        "/content/Test/assets/audio/bgm/bgm_action_combat.wav",
        "/content/Test/assets/audio/bgm/bgm_heroic_theme.wav",
    ]
    bgm = next((p for p in bgm_candidates if os.path.exists(p)), None)

    cmd = [ffmpeg, "-y", "-i", raw_mp4, "-i", sfx_wav]
    filters = ["[1:a]volume=0.85[sfx]"]
    mix_labels = ["[sfx]"]
    idx = 2

    if bgm:
        cmd += ["-stream_loop", "-1", "-i", bgm]
        filters.append(f"[{idx}:a]volume=0.22[bgm]")
        mix_labels.append("[bgm]")
        idx += 1

    for t_sec, mp3 in dialogue_items:
        cmd += ["-i", mp3]
        delay_ms = int(t_sec * 1000)
        filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume=1.35[d{idx}]")
        mix_labels.append(f"[d{idx}]")
        idx += 1

    filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[aout]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", final_mp4]
    subprocess.run(cmd, check=True)
    print("-> Đã ghép xong toàn bộ âm thanh võ thuật 1 phút:", final_mp4, flush=True)


def main():
    if "--blender-render" in sys.argv:
        idx = sys.argv.index("--blender-render")
        run_blender_mocap_fight(sys.argv[idx + 1], sys.argv[idx + 2])
        return

    work_dir = "/content/Test/assets/models_3d"
    out_dir = "/content/Test/build_output"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    manifest = download_mixamo_assets(work_dir)
    manifest_path = os.path.join(work_dir, "mixamo_manifest.json")
    raw_mp4 = os.path.join(out_dir, "demo_mocap_1min_raw.mp4")
    sfx_wav = os.path.join(out_dir, "demo_mocap_sfx.wav")
    final_mp4 = os.path.join(out_dir, "demo_sentai_3d_scene.mp4")

    cmd = [
        "/content/blender/blender", "--background",
        "--python", os.path.abspath(__file__), "--",
        "--blender-render", manifest_path, raw_mp4,
    ]
    subprocess.run(cmd, check=True)

    hits_json = raw_mp4 + ".hits.json"
    hit_frames = json.load(open(hits_json, "r", encoding="utf-8")) if os.path.exists(hits_json) else []
    generate_impact_sfx_track(sfx_wav, hit_frames, duration_sec=60.0)
    dialogue_items = synth_dialogue_lines(out_dir)
    mux_full_audio(raw_mp4, sfx_wav, dialogue_items, final_mp4)
    print(f"HOÀN TẤT VIDEO VÕ THUẬT MOCAP THẬT 1 PHÚT: {final_mp4} ({os.path.getsize(final_mp4)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
