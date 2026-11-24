r"""
Sentai 3D Director — Cảnh Rượt Đuổi Parkour Nhảy Qua Các Tòa Nhà Đêm (30 giây, 720p 24fps)
- 2 nhân vật NGƯỜI THẬT Mixamo (chuẩn hóa sạch mọi tiền tố xương mixamorig\d*: -> 100% cả 2 cùng cử động!)
- Motion Capture Parkour thật từ Mixamo (Hugging Face): Running/Fast_Run + Running_Jump + Forward_Roll
- 4 nóc tòa nhà cao tầng có khe hở 3.5–4.0m giữa bầu trời đêm thành phố neon + quỹ đạo nhảy parabol thật
- 6 góc máy rượt đuổi điện ảnh (Drone bám ngang, Góc nhảy vọt qua đầu camera, Góc đuổi sau lưng) + thoại Việt & SFX
"""
import asyncio
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import wave


def download_parkour_assets(work_dir):
    """Tự động tải 2 nhân vật người thật + bộ Motion Capture Parkour (chạy, nhảy, lộn nhào) từ Hugging Face Mixamo."""
    from huggingface_hub import HfApi, hf_hub_download

    manifest_path = os.path.join(work_dir, "parkour_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            os.path.exists(data.get("hero_fbx", ""))
            and os.path.exists(data.get("enemy_fbx", ""))
            and all(os.path.exists(p) for p in data.get("anims", {}).values())
            and len(data.get("anims", {})) >= 3
        ):
            print("Dùng lại bộ nhân vật & Motion Capture Parkour đã tải sẵn!", flush=True)
            return data

    api = HfApi()
    repo_id = None
    files = []
    for cand in ("Linzhan/Mixamo-Animations-Characters", "tanish434/Mixamo-Animations-Characters"):
        try:
            print(f"Đang quét kho Mixamo Parkour trên Hugging Face ({cand})...", flush=True)
            files = api.list_repo_files(repo_id=cand, repo_type="dataset")
            if files:
                repo_id = cand
                break
        except Exception as e:
            print(f"  Bỏ qua {cand}: {e}", flush=True)

    if not repo_id or not files:
        raise RuntimeError("Không truy cập được kho Mixamo trên Hugging Face.")

    char_fbxs = [f for f in files if f.lower().endswith(".fbx") and ("character_refined/" in f.lower() or f.lower().startswith("character/"))]
    if not char_fbxs:
        char_fbxs = [f for f in files if f.lower().endswith(".fbx") and "character" in f.lower()]

    def pick_char(preferred_names, exclude=None):
        for name in preferred_names:
            for f in char_fbxs:
                if f == exclude:
                    continue
                if name.lower() in os.path.basename(f).lower():
                    return f
        for f in char_fbxs:
            if f != exclude:
                return f
        return None

    # Chọn 2 nhân vật người nam trưởng thành cân đối (ưu tiên character_refined trước)
    char_fbxs.sort(key=lambda x: (0 if "character_refined" in x.lower() else 1, x))
    hero_remote = pick_char(["david", "douglas", "leonard", "adam", "james", "brian", "malcolm", "bryce", "ch14", "ch17", "ch31"])
    enemy_remote = pick_char(["swat", "soldier", "malcolm", "douglas", "leonard", "brian", "josh", "pete", "ch15", "ch33"], exclude=hero_remote)

    anim_fbxs = [f for f in files if f.lower().endswith(".fbx") and "animation" in f.lower()]
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
        for kw in keywords:
            k = kw.lower()
            for f in anim_fbxs:
                base = os.path.basename(f).lower().replace("_", " ").replace("-", " ")
                if k in base:
                    return f
        return None

    used = set()
    anim_queries = {
        "run": ["fast run", "sprinting", "running", "run forward", "jog"],
        "jump": ["running jump", "jump over", "big jump", "jumping", "jump"],
        "roll": ["sprinting forward roll", "falling to roll", "stand to roll", "roll", "landing"],
    }
    selected_anims_remote = {}
    for slot, kws in anim_queries.items():
        found = pick_anim(kws, used)
        if found:
            selected_anims_remote[slot] = found

    print(f"Hero (người đuổi):  {hero_remote}", flush=True)
    print(f"Enemy (người chạy): {enemy_remote}", flush=True)
    print("Động tác Parkour Motion Capture thật:", flush=True)
    for k, v in selected_anims_remote.items():
        print(f"  - {k:6s}: {os.path.basename(v)}", flush=True)

    local_dir = os.path.join(work_dir, "mixamo_cache")
    os.makedirs(local_dir, exist_ok=True)

    def fetch(remote_path):
        return hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="dataset", local_dir=local_dir)

    manifest = {
        "hero_fbx": fetch(hero_remote),
        "enemy_fbx": fetch(enemy_remote),
        "anims": {k: fetch(v) for k, v in selected_anims_remote.items()},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _norm_mixamo_bone(name):
    """Chuẩn hóa MỌI biến thể tên xương Mixamo (mixamorig1:Hips, mixamorig9:Spine, Hips -> mixamorig:Hips)."""
    clean = re.sub(r"^mixamorig\d*:", "", name)
    return "mixamorig:" + clean


def run_blender_rooftop_chase(manifest_path, out_raw_mp4):
    import bpy
    from mathutils import Vector

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    FPS = 24
    TOTAL_FRAMES = 30 * FPS  # 720 frames = 30 giây tốc độ cao, render siêu nhanh!
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

    # --- 1. DỰNG THÀNH PHỐ ĐÊM 3D VỚI 4 NÓC TÒA NHÀ CAO TẦNG CÓ KHE HỞ ĐỂ NHẢY QUA ---
    mat_roof = make_pbr_mat("Roof_Concrete", (0.045, 0.05, 0.065, 1.0), metallic=0.15, roughness=0.35)
    mat_wall = make_pbr_mat("Building_Wall", (0.03, 0.035, 0.05, 1.0), metallic=0.35, roughness=0.45)
    mat_ledge = make_pbr_mat("Roof_Ledge", (0.10, 0.11, 0.14, 1.0), metallic=0.4, roughness=0.35)
    mat_neon_red = make_pbr_mat("Neon_Red", (1.0, 0.08, 0.05, 1.0), 0.1, 0.2, (1.0, 0.12, 0.05, 1.0), 14.0)
    mat_neon_cyan = make_pbr_mat("Neon_Cyan", (0.05, 0.78, 1.0, 1.0), 0.1, 0.2, (0.06, 0.78, 1.0, 1.0), 12.0)
    mat_neon_gold = make_pbr_mat("Neon_Gold", (1.0, 0.68, 0.10, 1.0), 0.1, 0.2, (1.0, 0.65, 0.12, 1.0), 10.0)
    mat_win = make_pbr_mat("City_Window", (0.85, 0.90, 1.0, 1.0), 0.1, 0.1, (0.75, 0.85, 1.0, 1.0), 4.5)

    # 4 tòa nhà chính dọc trục +Y mà 2 nhân vật chạy & nhảy qua: (y_start, y_end, roof_z)
    rooftops = [
        (-4.0, 14.0, 0.0),      # Tòa 1: dài 18m, cao Z=0.0m
        (18.5, 36.0, 1.1),      # Khe hở 4.5m -> Tòa 2 cao hơn (+1.1m)
        (41.0, 58.0, -0.4),     # Khe hở 5.0m -> Tòa 3 thấp hơn (-0.4m, cú nhảy lao xuống cực đã!)
        (63.0, 92.0, 0.8),      # Khe hở 5.0m -> Tòa 4 cao lên (+0.8m)
    ]

    for idx, (y0, y1, rz) in enumerate(rooftops):
        length = y1 - y0
        cy = 0.5 * (y0 + y1)
        # Thân tòa nhà cao 30m đâm sâu xuống vực thành phố
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, cy, rz - 15.0))
        b = bpy.context.active_object
        b.scale = (9.0, length, 30.0)
        bpy.ops.object.transform_apply(scale=True)
        b.data.materials.append(mat_roof)

        # Dải đèn neon ở mép nhảy đầu/cuối mỗi tòa nhà (nhìn cực rõ điểm giậm nhảy & tiếp đất!)
        for ey, nmat in [(y0 + 0.15, mat_neon_cyan), (y1 - 0.15, mat_neon_red)]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, ey, rz + 0.06))
            edge = bpy.context.active_object
            edge.scale = (8.8, 0.22, 0.12)
            bpy.ops.object.transform_apply(scale=True)
            edge.data.materials.append(nmat)

        # Khối cục nóng điều hòa / bồn nước / biển hiệu neon 2 bên mép sân thượng (tạo cảm giác tốc độ lướt qua!)
        for sx in (-3.6, 3.6):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, cy, rz + 0.8))
            prop = bpy.context.active_object
            prop.scale = (1.3, 2.4, 1.6)
            bpy.ops.object.transform_apply(scale=True)
            prop.data.materials.append(mat_ledge)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, y1 - 2.2, rz + 2.4))
            sign = bpy.context.active_object
            sign.scale = (0.18, 2.2, 1.4)
            bpy.ops.object.transform_apply(scale=True)
            sign.data.materials.append(mat_neon_red if (idx % 2 == 0) else mat_neon_gold)

    # Thành phố cao ốc 2 bên (trái X=-14..-28, phải X=+14..+28) có cửa sổ sáng rực tạo chiều sâu choáng ngợp
    rng = random.Random(2026)
    for by in range(-10, 115, 11):
        for side in (-1, 1):
            bx = side * rng.uniform(14.0, 26.0)
            bh = rng.uniform(28.0, 52.0)
            bw = rng.uniform(6.0, 10.0)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bh / 2.0 - 22.0))
            tower = bpy.context.active_object
            tower.scale = (bw, bw, bh)
            bpy.ops.object.transform_apply(scale=True)
            tower.data.materials.append(mat_wall)

            # Dải đèn cửa sổ ngang phát sáng trên mỗi cao ốc
            for wz in (-6.0, 0.0, 6.0, 12.0):
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx - side * (bw * 0.51), by, wz))
                win = bpy.context.active_object
                win.scale = (0.12, bw * 0.72, 0.55)
                bpy.ops.object.transform_apply(scale=True)
                win.data.materials.append(mat_win if rng.random() > 0.35 else mat_neon_cyan)

    # --- 2. NẠP 3 ĐỘNG TÁC PARKOUR THẬT (RUN / JUMP / ROLL) & CHUẨN HÓA TÊN XƯƠNG 100% ---
    mocap_actions = {}
    for clip_key, fbx_path in manifest["anims"].items():
        before_objs = set(bpy.context.scene.objects)
        before_acts = set(bpy.data.actions)
        bpy.ops.import_scene.fbx(filepath=fbx_path, ignore_leaf_bones=True)
        new_objs = [o for o in bpy.context.scene.objects if o not in before_objs]
        new_acts = [a for a in bpy.data.actions if a not in before_acts]
        for o in new_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        if not new_acts:
            continue

        act = new_acts[0]
        act.name = f"Parkour_{clip_key}"
        act.use_fake_user = True

        # Chuẩn hóa tên xương trong mọi fcurve về đúng "mixamorig:<Bone>" + xóa scale + khóa trượt ngang in-place!
        for fc in list(act.fcurves):
            if fc.data_path.endswith("scale"):
                act.fcurves.remove(fc)
                continue
            m = re.match(r'^pose\.bones\["([^"]+)"\](.*)$', fc.data_path)
            if m:
                fc.data_path = f'pose.bones["{_norm_mixamo_bone(m.group(1))}"]{m.group(2)}'
            # Giữ chuyển động tại chỗ (in-place) theo X/Z để quỹ đạo chạy & nhảy qua tòa nhà do root_empty điều khiển chính xác!
            if "hips" in fc.data_path.lower() and fc.data_path.endswith("location") and fc.array_index in (0, 2):
                if fc.keyframe_points:
                    base_v = fc.keyframe_points[0].co.y
                    for kp in fc.keyframe_points:
                        kp.co.y = base_v
                        kp.handle_left.y = base_v
                        kp.handle_right.y = base_v

        f_len = max(12, int(round(act.frame_range[1] - act.frame_range[0])))
        mocap_actions[clip_key] = (act, f_len)
        print(f"Loaded Parkour Mocap: {act.name} ({f_len} frames)", flush=True)

    # --- 3. NẠP 2 NHÂN VẬT NGƯỜI THẬT MIXAMO & CHUẨN HÓA XƯƠNG + CHIỀU CAO 1.78M ---
    def load_runner(name, fbx_path):
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

        # Chuẩn hóa TẤT CẢ tên xương (kể cả mixamorig1:, mixamorig9:, hay không tiền tố) -> mixamorig:<Bone>
        for b in arm.data.bones:
            b.name = _norm_mixamo_bone(b.name)

        bpy.context.view_layer.update()
        pts = []
        for m in meshes:
            pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
        min_z = min(v.z for v in pts)
        max_z = max(v.z for v in pts)
        raw_h = max(0.05, max_z - min_z)
        sc = 1.78 / raw_h

        root_empty = bpy.data.objects.new(f"{name}_Root", None)
        scene.collection.objects.link(root_empty)
        for r in roots:
            r.parent = root_empty
            r.location.z -= min_z
        root_empty.scale = (sc, sc, sc)
        # Mixamo FBX mặc định mặt nhìn về -Y -> xoay 180 độ để chạy thẳng dọc trục +Y qua 4 tòa nhà!
        root_empty.rotation_euler = (0.0, 0.0, math.radians(180.0))

        for m in meshes:
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.select_all(action="DESELECT")
            m.select_set(True)
            bpy.ops.object.shade_smooth()

        if not arm.animation_data:
            arm.animation_data_create()
        print(f"[{name}] Ready! height={raw_h:.2f}m->1.78m, bones={len(arm.data.bones)} (all normalized to mixamorig:*)", flush=True)
        return root_empty, arm

    hero_root, hero_arm = load_runner("HoaDo", manifest["hero_fbx"])
    enemy_root, enemy_arm = load_runner("MacKhieu", manifest["enemy_fbx"])

    # --- 4. TÍNH QUỸ ĐẠO CHẠY + NHẢY PARABOL QUA 3 KHE TÒA NHÀ & ĐỒNG BỘ NLA (RUN -> JUMP -> ROLL -> RUN) ---
    def roof_z_and_state(y):
        """Trả về (z_height, state) theo vị trí dọc trục Y: state in ('run', 'jump', 'roll')."""
        for i, (y0, y1, rz) in enumerate(rooftops):
            if y0 <= y <= y1:
                # 1.8m ngay sau khi tiếp đất từ khe nhảy trước -> lộn nhào parkour (roll)!
                if i > 0 and (y - y0) < 2.2:
                    return rz, "roll"
                return rz, "run"
            # Đang bay qua khe hở giữa tòa i và tòa i+1!
            if i < len(rooftops) - 1:
                ny0, _, nrz = rooftops[i + 1]
                if y1 < y < ny0:
                    t = (y - y1) / (ny0 - y1)  # 0..1
                    # Đường parabol nhảy cao vọt thêm 1.65m giữa không trung!
                    z_base = rz * (1.0 - t) + nrz * t
                    z_arc = 4.0 * 1.65 * t * (1.0 - t)
                    return z_base + z_arc, "jump"
        return rooftops[-1][2], "run"

    def animate_parkour_runner(root_obj, arm, x_lane, y_start, speed_mps):
        """Di chuyển nhân vật dọc +Y với vận tốc thật (~2.8 m/s -> 84m trong 30s) & đồng bộ động tác Run/Jump/Roll."""
        jump_frames = []
        state_timeline = []
        for fr in range(1, TOTAL_FRAMES + 1):
            t_sec = (fr - 1) / float(FPS)
            y = y_start + speed_mps * t_sec
            z, st = roof_z_and_state(y)
            # Lắc nhẹ ngang tạo cảm giác chạy rượt đuổi tự nhiên
            x = x_lane + 0.15 * math.sin(t_sec * 1.1)
            root_obj.location = (x, y, z)
            root_obj.keyframe_insert(data_path="location", frame=fr)
            state_timeline.append((fr, st))

        # Chuyển đổi state_timeline thành các đoạn (start_fr, end_fr, clip_key) để đặt NLA track A/B mượt mà!
        segments = []
        cur_st = state_timeline[0][1]
        seg_start = 1
        for fr, st in state_timeline[1:]:
            if st != cur_st:
                segments.append((seg_start, fr - 1, cur_st))
                if cur_st == "jump":
                    jump_frames.append((seg_start, fr - 1))
                cur_st = st
                seg_start = fr
        segments.append((seg_start, TOTAL_FRAMES, cur_st))

        track_a = arm.animation_data.nla_tracks.new()
        track_a.name = "Parkour_A"
        track_b = arm.animation_data.nla_tracks.new()
        track_b.name = "Parkour_B"
        tracks = [track_a, track_b]

        for idx_s, (f0, f1, st) in enumerate(segments):
            clip_key = st if st in mocap_actions else "run"
            act, f_len = mocap_actions[clip_key]
            tr = tracks[idx_s % 2]
            start_f = f0
            if tr.strips:
                start_f = max(start_f, int(math.ceil(tr.strips[-1].frame_end)) + 1)
            if start_f > TOTAL_FRAMES:
                break
            strip = tr.strips.new(f"{clip_key}_{start_f}", start=start_f, action=act)
            strip.blend_type = "REPLACE"
            strip.extrapolation = "NOTHING" if idx_s < len(segments) - 1 else "HOLD_FORWARD"
            dur = max(6, f1 - start_f + 4)
            if clip_key == "run":
                strip.repeat = max(1.0, dur / max(1.0, float(f_len)))
            else:
                # Co giãn nhẹ tốc độ clip jump/roll vừa khít đúng thời gian bay qua khe tòa nhà!
                strip.scale = max(0.45, min(1.8, dur / max(1.0, float(f_len))))
            strip.blend_in = min(5, max(1, dur // 5))
            strip.blend_out = 0
        return jump_frames

    # Mạc Khiêu chạy trước 3.8m (y_start = 2.0m), Hỏa Đỏ chạy đuổi theo sát phía sau (y_start = -1.8m), vận tốc 2.75 m/s!
    enemy_jumps = animate_parkour_runner(enemy_root, enemy_arm, x_lane=0.45, y_start=2.0, speed_mps=2.72)
    hero_jumps = animate_parkour_runner(hero_root, hero_arm, x_lane=-0.35, y_start=-1.8, speed_mps=2.82)

    with open(out_raw_mp4 + ".jumps.json", "w", encoding="utf-8") as jf:
        json.dump({"hero_jumps": hero_jumps, "enemy_jumps": enemy_jumps}, jf)

    # --- 5. ÁNH SÁNG THÀNH PHỐ ĐÊM ĐIỆN ẢNH ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.015, 0.022, 0.045, 1.0)
        bg.inputs[1].default_value = 0.85

    # Đèn mặt trăng / ánh sáng đêm bao trùm toàn bộ 4 tòa nhà + 2 đèn bám theo 2 nhân vật để mặt & đồ luôn sáng rõ!
    bpy.ops.object.light_add(type="SUN", location=(10, 20, 40), rotation=(math.radians(48), math.radians(18), math.radians(-30)))
    sun = bpy.context.active_object
    sun.data.energy = 2.2
    sun.data.color = (0.78, 0.86, 1.0)

    def add_follow_light(name, parent, offset, energy, color):
        bpy.ops.object.light_add(type="POINT", location=(0, 0, 0))
        l = bpy.context.active_object
        l.name = name
        l.parent = parent
        l.location = offset
        l.data.energy = energy
        l.data.color = color
        l.data.shadow_soft_size = 1.2

    add_follow_light("Hero_KeyLight", hero_root, (-1.8, 1.6, 2.4), 260.0, (1.0, 0.85, 0.72))
    add_follow_light("Hero_RimRed", hero_root, (1.4, -1.8, 2.0), 220.0, (1.0, 0.20, 0.08))
    add_follow_light("Enemy_KeyLight", enemy_root, (1.8, 1.6, 2.4), 240.0, (0.75, 0.90, 1.0))

    # --- 6. HỆ THỐNG 6 GÓC MÁY RƯỢT ĐUỔI ĐIỆN ẢNH (DRONE BÁM NGANG + NHẢY QUA ĐẦU + CHASE CAM) ---
    chase_mid = bpy.data.objects.new("ChaseMidTarget", None)
    scene.collection.objects.link(chase_mid)
    for fr in range(1, TOTAL_FRAMES + 1, 6):
        scene.frame_set(fr)
        hl = hero_root.location
        el = enemy_root.location
        chase_mid.location = (0.5 * (hl.x + el.x), 0.5 * (hl.y + el.y), 0.5 * (hl.z + el.z) + 1.15)
        chase_mid.keyframe_insert(data_path="location", frame=fr)

    hero_tgt = bpy.data.objects.new("HeroTarget", None)
    scene.collection.objects.link(hero_tgt)
    hero_tgt.parent = hero_root
    hero_tgt.location = (0.0, 0.0, 1.35)

    def make_follow_cam(name, parent_obj, offset_start, offset_end, f1, f2, target_obj, lens=32.0):
        bpy.ops.object.camera_add(location=(0, 0, 0))
        c = bpy.context.active_object
        c.name = name
        c.data.lens = lens
        c.parent = parent_obj
        c.location = offset_start
        c.keyframe_insert(data_path="location", frame=f1)
        c.location = offset_end
        c.keyframe_insert(data_path="location", frame=f2)
        con = c.constraints.new(type="TRACK_TO")
        con.target = target_obj
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        return c

    def make_static_gap_cam(name, loc_start, loc_end, f1, f2, target_obj, lens=28.0):
        bpy.ops.object.camera_add(location=loc_start)
        c = bpy.context.active_object
        c.name = name
        c.data.lens = lens
        c.location = loc_start
        c.keyframe_insert(data_path="location", frame=f1)
        c.location = loc_end
        c.keyframe_insert(data_path="location", frame=f2)
        con = c.constraints.new(type="TRACK_TO")
        con.target = target_obj
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        return c

    # 6 góc máy bám nhịp 3 cú nhảy qua khe tòa nhà:
    cam_schedule = [
        # Shot 1 (0–5s): Góc rượt đuổi sau vai Hỏa Đỏ đang tăng tốc đuổi Mạc Khiêu trên Tòa 1
        (1, make_follow_cam("C1_ChaseBehind", hero_root, (-1.8, -3.2, 2.0), (-1.2, -2.5, 1.7), 1, 120, chase_mid, 30.0)),
        # Shot 2 (5–10s): Góc thấp ngay dưới khe hở Tòa 1->2 (2 người phóng vọt qua ngay trên đầu camera!)
        (121, make_static_gap_cam("C2_Gap1_LowLeap", (5.5, 16.2, -0.4), (4.5, 17.0, 1.4), 121, 240, chase_mid, 28.0)),
        # Shot 3 (10–15s): Drone bay song song bên trái trên Tòa 2 khoe cả thành phố đêm & 2 người tăng tốc
        (241, make_follow_cam("C3_DroneSideL", chase_mid, (-6.2, 1.0, 1.6), (-5.5, 2.5, 2.2), 241, 360, chase_mid, 32.0)),
        # Shot 4 (15–20s): Khe hở Tòa 2->3 (cú nhảy lao từ cao xuống thấp cực kịch tính!)
        (361, make_static_gap_cam("C4_Gap2_DropLeap", (-6.0, 38.5, 0.2), (-4.5, 40.5, 1.8), 361, 480, chase_mid, 26.0)),
        # Shot 5 (20–25s): Góc quay ngược từ phía trước (nhìn thẳng mặt Hỏa Đỏ đang đuổi sát nút trên Tòa 3!)
        (481, make_follow_cam("C5_FrontChase", hero_root, (1.6, 3.8, 1.4), (-1.2, 3.0, 1.5), 481, 600, hero_tgt, 38.0)),
        # Shot 6 (25–30s): Khe hở cuối Tòa 3->4 (góc rộng điện ảnh cả 2 phóng vọt qua ánh đèn thành phố!)
        (601, make_static_gap_cam("C6_Gap3_EpicWide", (7.2, 60.5, 1.8), (6.0, 64.0, 3.2), 601, 720, chase_mid, 28.0)),
    ]
    scene.camera = cam_schedule[0][1]
    for fstart, cam_obj in cam_schedule:
        mk = scene.timeline_markers.new(cam_obj.name, frame=fstart)
        mk.camera = cam_obj

    # --- 7. RENDER CYCLES GPU SIÊU TỐC (720 FRAMES @ 720p, 24FPS ĐÚNG 30 GIÂY) ---
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
    print(f"GPU đang dùng: {', '.join(gpu_names) if gpu_names else 'CPU'} | 720 frames @ 24fps (30s)", flush=True)

    # Ép lại FPS = 24 SAU KHI import FBX
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = out_raw_mp4
    os.makedirs(os.path.dirname(out_raw_mp4), exist_ok=True)
    bpy.ops.render.render(animation=True)
    print("-> Render Blender Parkour hoàn tất:", out_raw_mp4, flush=True)


def generate_parkour_sfx(out_wav, jump_info, duration_sec=30.0):
    """Tạo tiếng gió vút (whoosh) lúc nhảy qua tòa nhà & tiếng tiếp đất (thud) khớp từng cú nhảy."""
    sr = 44100
    n_samples = int(sr * duration_sec)
    buf = [0.0] * n_samples
    rng = random.Random(2026)

    all_jumps = jump_info.get("hero_jumps", []) + jump_info.get("enemy_jumps", [])
    for f0, f1 in all_jumps:
        # Tiếng gió vút lúc giậm nhảy
        s0 = int((f0 / 24.0) * sr)
        wlen = int(0.35 * sr)
        for i in range(wlen):
            idx = s0 + i
            if idx >= n_samples:
                break
            t = i / sr
            env = math.sin(math.pi * (i / wlen)) * 0.45
            buf[idx] = max(-0.95, min(0.95, buf[idx] + (rng.random() * 2 - 1) * env))
        # Tiếng tiếp đất mạnh
        s1 = int((f1 / 24.0) * sr)
        tlen = int(0.20 * sr)
        for i in range(tlen):
            idx = s1 + i
            if idx >= n_samples:
                break
            t = i / sr
            thump = math.sin(2 * math.pi * (95.0 * math.exp(-t * 16.0)) * t) * math.exp(-t * 22.0) * 0.8
            buf[idx] = max(-0.95, min(0.95, buf[idx] + thump))

    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(struct.pack("<h", int(s * 32000)) for s in buf))


def synth_chase_dialogue(tmp_dir):
    lines = [
        (1.2, "vi-VN-NamMinhNeural", "+8%", "Mạc Khiêu! Đứng lại ngay!"),
        (6.2, "vi-VN-NamMinhNeural", "+2%", "Đuổi kịp ta qua dãy cao ốc này đi rồi nói!"),
        (14.5, "vi-VN-NamMinhNeural", "+10%", "Đừng hòng chạy thoát!"),
        (23.5, "vi-VN-NamMinhNeural", "+10%", "Ta bắt kịp ngươi rồi!"),
    ]
    out_items = []
    try:
        import edge_tts

        async def _gen():
            for i, (t_sec, voice, rate, text) in enumerate(lines):
                mp3 = os.path.join(tmp_dir, f"chase_line_{i:02d}.mp3")
                try:
                    comm = edge_tts.Communicate(text, voice=voice, rate=rate)
                    await comm.save(mp3)
                except Exception:
                    try:
                        comm = edge_tts.Communicate(text, voice=voice)
                        await comm.save(mp3)
                    except Exception:
                        continue
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
    filters = ["[1:a]volume=0.90[sfx]"]
    mix_labels = ["[sfx]"]
    idx = 2

    if bgm:
        cmd += ["-stream_loop", "-1", "-i", bgm]
        filters.append(f"[{idx}:a]volume=0.25[bgm]")
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
    print("-> Đã ghép xong âm thanh rượt đuổi Parkour 30s:", final_mp4, flush=True)


def main():
    if "--blender-render" in sys.argv:
        idx = sys.argv.index("--blender-render")
        run_blender_rooftop_chase(sys.argv[idx + 1], sys.argv[idx + 2])
        return

    work_dir = "/content/Test/assets/models_3d"
    out_dir = "/content/Test/build_output"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    manifest = download_parkour_assets(work_dir)
    manifest_path = os.path.join(work_dir, "parkour_manifest.json")
    raw_mp4 = os.path.join(out_dir, "demo_parkour_chase_raw.mp4")
    sfx_wav = os.path.join(out_dir, "demo_parkour_sfx.wav")
    final_mp4 = os.path.join(out_dir, "demo_sentai_3d_scene.mp4")

    cmd = [
        "/content/blender/blender", "--background",
        "--python", os.path.abspath(__file__), "--",
        "--blender-render", manifest_path, raw_mp4,
    ]
    subprocess.run(cmd, check=True)

    jumps_json = raw_mp4 + ".jumps.json"
    jump_info = json.load(open(jumps_json, "r", encoding="utf-8")) if os.path.exists(jumps_json) else {}
    generate_parkour_sfx(sfx_wav, jump_info, duration_sec=30.0)
    dialogue_items = synth_chase_dialogue(out_dir)
    mux_full_audio(raw_mp4, sfx_wav, dialogue_items, final_mp4)
    print(f"HOÀN TẤT VIDEO RƯỢT ĐUỔI PARKOUR 3D: {final_mp4} ({os.path.getsize(final_mp4)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
