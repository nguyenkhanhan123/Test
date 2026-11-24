"""
Sentai 3D Director — Cảnh Võ Thuật 1 Phút (60 giây, 720p 24fps, Người thật chưa biến thân)
- 2 nhân vật NGƯỜI THẬT 3D chuẩn Rig (có khuôn mặt, tóc, trang phục thật — Ready Player Me Male + Soldier)
- 60 giây biên đạo võ thuật 6 hồi: Đối đầu -> Đấm Jab/Cross -> Đá tống & Đá xoay -> Phản công -> Liên hoàn Hỏa Quyền -> Kết thúc
- 8 góc máy hành động điện ảnh tự động cắt cảnh + hiệu ứng tia lửa va chạm (Impact Flash/Ring)
- Ghép thoại tiếng Việt 2 nhân vật (edge-tts) + tiếng đấm/đá (SFX) đồng bộ đúng khung hình + nhạc nền chiến đấu
"""
import asyncio
import math
import os
import shutil
import subprocess
import sys
import urllib.request
import wave
import struct


def run_blender_martial_arts(hero_glb, enemy_glb, out_raw_mp4):
    import bpy
    from mathutils import Euler, Vector

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

    def tint_material(mat, rgba, factor=0.78):
        if not mat or not mat.use_nodes or not mat.node_tree:
            return
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf:
            return
        base_sock = bsdf.inputs.get("Base Color")
        if base_sock is None:
            return
        if base_sock.is_linked:
            orig_link = base_sock.links[0]
            from_sock = orig_link.from_socket
            nt.links.remove(orig_link)
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = factor
            mix.inputs["B"].default_value = rgba
            nt.links.new(from_sock, mix.inputs["A"])
            nt.links.new(mix.outputs["Result"], base_sock)
        else:
            base_sock.default_value = rgba

    # --- 1. BỐI CẢNH SÂN THƯỢNG / ĐẤU TRƯỜNG ĐÊM 3D THẬT ---
    mat_floor = make_pbr_mat("Rooftop_Floor", (0.03, 0.035, 0.045, 1.0), metallic=0.18, roughness=0.28)
    mat_concrete = make_pbr_mat("Concrete_Wall", (0.07, 0.075, 0.09, 1.0), metallic=0.15, roughness=0.65)
    mat_neon_red = make_pbr_mat("Neon_Red", (1.0, 0.10, 0.05, 1.0), metallic=0.1, roughness=0.2, emission=(1.0, 0.14, 0.05, 1.0), em_strength=10.0)
    mat_neon_blue = make_pbr_mat("Neon_Blue", (0.08, 0.65, 1.0, 1.0), metallic=0.1, roughness=0.2, emission=(0.08, 0.68, 1.0, 1.0), em_strength=9.0)
    mat_spark = make_pbr_mat("Impact_Spark", (1.0, 0.45, 0.08, 1.0), metallic=0.0, roughness=0.1, emission=(1.0, 0.35, 0.05, 1.0), em_strength=28.0)

    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.data.materials.append(mat_floor)

    # Vạch sân đấu & hàng trụ đèn sân thượng thành phố về đêm
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

    # Đường chân trời tòa nhà phía sau tạo chiều sâu thành phố
    for bx, by, bh in [(-7, 10, 8), (-3, 12, 11), (2, 11, 9), (6.5, 10, 12)]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bh / 2.0))
        b = bpy.context.active_object
        b.scale = (2.6, 2.2, bh)
        bpy.ops.object.transform_apply(scale=True)
        b.data.materials.append(mat_concrete)

    # --- 2. NẠP 2 NHÂN VẬT NGƯỜI THẬT 3D (CÓ MẶT, TÓC, QUẦN ÁO THẬT) ---
    def load_human_fighter(name, glb_path, top_tint=None, bottom_tint=None):
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        new_objs = [o for o in bpy.context.scene.objects if o not in before]
        arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
        meshes = [o for o in new_objs if o.type == "MESH"]
        roots = [o for o in new_objs if o.parent is None]
        if arm is None:
            raise RuntimeError(f"No armature in {glb_path}")

        # Xóa animation mặc định (nếu có) để biên đạo võ thuật trực tiếp sạch 100%
        if arm.animation_data:
            arm.animation_data.action = None
            for tr in list(arm.animation_data.nla_tracks):
                arm.animation_data.nla_tracks.remove(tr)

        # Chuẩn hóa chiều cao đúng 1.78m đứng trên Z=0
        bpy.context.view_layer.update()
        pts = []
        for m in meshes:
            pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
        min_z = min(v.z for v in pts)
        max_z = max(v.z for v in pts)
        h = max(0.2, max_z - min_z)
        sc = 1.78 / h

        root_empty = bpy.data.objects.new(f"{name}_Root", None)
        scene.collection.objects.link(root_empty)
        for r in roots:
            r.scale = (r.scale.x * sc, r.scale.y * sc, r.scale.z * sc)
            r.location.z -= min_z * sc
            r.parent = root_empty

        # Làm mịn lưới & nhuộm màu trang phục (TUYỆT ĐỐI giữ nguyên mặt, mắt, da, tóc thật!)
        for m in meshes:
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.select_all(action="DESELECT")
            m.select_set(True)
            bpy.ops.object.shade_smooth()
            m_low = m.name.lower()
            for slot in m.material_slots:
                mat = slot.material
                if not mat:
                    continue
                low = mat.name.lower() + " " + m_low
                if any(k in low for k in ("head", "face", "eye", "skin", "teeth", "brow", "lash", "beard")):
                    continue
                if top_tint and any(k in low for k in ("top", "shirt", "jacket", "outfit_top", "coat", "body")):
                    tint_material(mat, top_tint, factor=0.78)
                elif bottom_tint and any(k in low for k in ("bottom", "pant", "trouser", "shoe", "footwear")):
                    tint_material(mat, bottom_tint, factor=0.82)

        # Kiểm tra hướng mặt (+Y hay -Y) để luôn đối mặt nhau chính xác
        pb = arm.pose.bones
        for b in pb:
            b.rotation_mode = "QUATERNION"

        def fb(*pats):
            for pat in pats:
                p = pat.lower()
                for b in pb:
                    bn = b.name.lower().replace("mixamorig:", "")
                    if bn == p:
                        return b
            for pat in pats:
                p = pat.lower()
                for b in pb:
                    bn = b.name.lower().replace("mixamorig:", "")
                    if p in bn:
                        return b
            return None

        bones = {
            "hips": fb("hips"),
            "spine": fb("spine"),
            "chest": fb("spine2", "spine1", "chest"),
            "neck": fb("neck"),
            "head": fb("head"),
            "ua_r": fb("rightarm", "rightupperarm"),
            "fa_r": fb("rightforearm", "rightlowerarm"),
            "hd_r": fb("righthand"),
            "ua_l": fb("leftarm", "leftupperarm"),
            "fa_l": fb("leftforearm", "leftlowerarm"),
            "hd_l": fb("lefthand"),
            "th_r": fb("rightupleg", "rightupperleg"),
            "kn_r": fb("rightleg", "rightlowerleg"),
            "ft_r": fb("rightfoot"),
            "th_l": fb("leftupleg", "leftupperleg"),
            "kn_l": fb("leftleg", "leftlowerleg"),
            "ft_l": fb("leftfoot"),
        }
        fingers_r, fingers_l = [], []
        for b in pb:
            low = b.name.lower()
            if any(f in low for f in ("index", "middle", "ring", "pinky", "little", "thumb")):
                is_th = "thumb" in low
                if "right" in low:
                    fingers_r.append((b, is_th))
                elif "left" in low:
                    fingers_l.append((b, is_th))

        bpy.context.view_layer.update()
        # Đo hướng mặt theo bàn chân (toe/foot hướng về phía trước)
        face_dir = 1.0  # +1 nghĩa là mặt đang hướng +Y trong local
        if bones["ft_r"] and bones["kn_r"]:
            ft_t = arm.matrix_world @ bones["ft_r"].bone.tail_local
            ft_h = arm.matrix_world @ bones["ft_r"].bone.head_local
            face_dir = 1.0 if (ft_t.y - ft_h.y) >= 0 else -1.0

        print(f"[{name}] Loaded human rig: {len(pb)} bones, fingers=({len(fingers_r)},{len(fingers_l)}), face_dir={face_dir}", flush=True)
        return {
            "name": name,
            "root": root_empty,
            "arm": arm,
            "bones": bones,
            "fingers_r": fingers_r,
            "fingers_l": fingers_l,
            "face_dir": face_dir,
        }

    # Hỏa Đỏ (Người thật trẻ trung, áo khoác đỏ-đen) vs Đối thủ (Người thật mặc đồ chiến thuật đen)
    hero = load_human_fighter("HoaDo", hero_glb, top_tint=(0.82, 0.08, 0.10, 1.0), bottom_tint=(0.08, 0.08, 0.10, 1.0))
    enemy = load_human_fighter("MacKhieu", enemy_glb, top_tint=(0.14, 0.12, 0.22, 1.0), bottom_tint=(0.06, 0.06, 0.08, 1.0))

    # Căn hướng đối mặt nhau trên trục Y: Hero đứng ở Y âm nhìn về +Y; Enemy đứng ở Y dương nhìn về -Y
    hero_base_yaw = 0.0 if hero["face_dir"] > 0 else 180.0
    enemy_base_yaw = 180.0 if enemy["face_dir"] > 0 else 0.0

    # --- 3. BỘ BIÊN ĐẠO VÕ THUẬT WORLD-SPACE (0% XUYÊN THÂN + NẮM ĐẤM 10 NGÓN) ---
    def key_bone_world(fighter, bkey, deg_xyz, frame):
        bone = fighter["bones"].get(bkey)
        if bone is None:
            return
        # Quy ước chung: +X = gập người/nhấc chân về trước, +Y = nghiêng/dạng ngang, +Z = xoay eo
        s = fighter["face_dir"]
        w_rot = Euler((math.radians(deg_xyz[0] * s), math.radians(deg_xyz[1] * s), math.radians(deg_xyz[2])), "XYZ").to_matrix()
        rest_m = bone.bone.matrix_local.to_3x3()
        local_m = rest_m.inverted() @ w_rot @ rest_m
        bone.rotation_quaternion = local_m.to_quaternion()
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    def key_fists(fighter, curl, frame):
        s = fighter["face_dir"]
        for flist, side in [(fighter["fingers_r"], -1), (fighter["fingers_l"], 1)]:
            for b, is_th in flist:
                ang = (22 * curl, 25 * curl * side, 20 * curl * side) if is_th else (0, 70 * curl * side, 0)
                w_rot = Euler((math.radians(ang[0] * s), math.radians(ang[1] * s), math.radians(ang[2])), "XYZ").to_matrix()
                rest_m = b.bone.matrix_local.to_3x3()
                b.rotation_quaternion = (rest_m.inverted() @ w_rot @ rest_m).to_quaternion()
                b.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Các tư thế võ thuật chuẩn (mọi tay đều mở rõ ngoài thân -> 0% xuyên áo!)
    POSES = {
        "stand": {
            "spine": (0, 0, 0), "chest": (0, 0, 0), "head": (0, 0, 0),
            "ua_r": (0, -52, 0), "fa_r": (0, 0, -10),
            "ua_l": (0, 52, 0), "fa_l": (0, 0, 10),
            "th_r": (0, 8, 0), "kn_r": (0, 0, 0), "th_l": (0, -8, 0), "kn_l": (0, 0, 0),
        },
        "guard": {
            "spine": (6, 0, -18), "chest": (6, 0, -16), "head": (-5, 0, 28),
            "ua_r": (0, -34, -38), "fa_r": (0, 0, -68),
            "ua_l": (0, 32, 48), "fa_l": (0, 0, 58),
            "th_r": (-18, 12, -12), "kn_r": (28, 0, 0),
            "th_l": (16, -12, 10), "kn_l": (22, 0, 0),
        },
        "jab_l": {
            "spine": (8, 0, 14), "chest": (6, 0, 20), "head": (-4, 0, -24),
            "ua_r": (0, -40, -24), "fa_r": (0, 0, -82),
            "ua_l": (0, 12, 82), "fa_l": (0, 0, 8),
            "th_r": (-14, 12, -10), "kn_r": (24, 0, 0),
            "th_l": (22, -10, 8), "kn_l": (26, 0, 0),
        },
        "cross_r": {
            "spine": (10, 0, 28), "chest": (8, 0, 26), "head": (-6, 0, -38),
            "ua_r": (0, -12, -84), "fa_r": (0, 0, -6),
            "ua_l": (0, 42, 20), "fa_l": (0, 0, 82),
            "th_r": (-22, 10, -14), "kn_r": (30, 0, 0),
            "th_l": (20, -12, 10), "kn_l": (24, 0, 0),
        },
        "uppercut_r": {
            "spine": (-6, 0, 26), "chest": (-8, 0, 24), "head": (4, 0, -30),
            "ua_r": (32, -26, -64), "fa_r": (0, 0, -76),
            "ua_l": (0, 44, 24), "fa_l": (0, 0, 78),
            "th_r": (-16, 12, -12), "kn_r": (24, 0, 0),
            "th_l": (16, -12, 10), "kn_l": (22, 0, 0),
        },
        "front_kick_r": {
            "spine": (-16, 0, -12), "chest": (-12, 0, -10), "head": (12, 0, 14),
            "ua_r": (0, -38, -28), "fa_r": (0, 0, -72),
            "ua_l": (0, 38, 34), "fa_l": (0, 0, 68),
            "th_r": (-86, 6, 0), "kn_r": (10, 0, 0),
            "th_l": (8, -6, 0), "kn_l": (14, 0, 0),
        },
        "roundhouse_r": {
            "spine": (-12, -26, 36), "chest": (-8, -18, 28), "head": (8, 16, -42),
            "ua_r": (0, -42, -20), "fa_r": (0, 0, -65),
            "ua_l": (0, 44, 30), "fa_l": (0, 0, 65),
            "th_r": (-72, 48, 20), "kn_r": (12, 0, 0),
            "th_l": (10, -8, -15), "kn_l": (16, 0, 0),
        },
        "block": {
            "spine": (-6, 0, -14), "chest": (-4, 0, -10), "head": (4, 0, 16),
            "ua_r": (0, -22, -56), "fa_r": (0, 0, -92),
            "ua_l": (0, 22, 56), "fa_l": (0, 0, 92),
            "th_r": (-16, 14, -8), "kn_r": (30, 0, 0),
            "th_l": (14, -14, 8), "kn_l": (26, 0, 0),
        },
        "hit_recoil": {
            "spine": (-22, 0, 18), "chest": (-18, 0, 16), "head": (-18, 0, 22),
            "ua_r": (0, -28, 14), "fa_r": (0, 0, -35),
            "ua_l": (0, 28, -14), "fa_l": (0, 0, 35),
            "th_r": (-10, 12, 0), "kn_r": (20, 0, 0),
            "th_l": (18, -12, 0), "kn_l": (24, 0, 0),
        },
        "kneel_defeat": {
            "spine": (26, 0, 0), "chest": (22, 0, 0), "head": (18, 0, 0),
            "ua_r": (0, -48, -18), "fa_r": (0, 0, -22),
            "ua_l": (0, 48, 18), "fa_l": (0, 0, 22),
            "th_r": (-76, 10, 0), "kn_r": (92, 0, 0),
            "th_l": (16, -12, 0), "kn_l": (95, 0, 0),
        },
    }

    def apply_pose(fighter, pose_name, frame, loc_xyz=None, yaw_offset=0.0, base_yaw=0.0, fist=1.0):
        bpy.context.view_layer.objects.active = fighter["arm"]
        bpy.ops.object.mode_set(mode="POSE")
        p = POSES[pose_name]
        for bk, deg in p.items():
            key_bone_world(fighter, bk, deg, frame)
        key_fists(fighter, fist, frame)
        bpy.ops.object.mode_set(mode="OBJECT")
        if loc_xyz is not None:
            fighter["root"].location = loc_xyz
            fighter["root"].keyframe_insert(data_path="location", frame=frame)
        fighter["root"].rotation_euler = (0, 0, math.radians(base_yaw + yaw_offset))
        fighter["root"].keyframe_insert(data_path="rotation_euler", frame=frame)

    # Vòng sáng va chạm (Impact Shockwave Ring) bung nổ đúng frame trúng đòn!
    bpy.ops.mesh.primitive_torus_add(major_radius=0.35, minor_radius=0.03, location=(0, 0, 1.35))
    spark_ring = bpy.context.active_object
    spark_ring.name = "ImpactShockwave"
    spark_ring.rotation_euler = (math.radians(90), 0, 0)
    spark_ring.data.materials.append(mat_spark)
    spark_ring.scale = (0.001, 0.001, 0.001)
    spark_ring.keyframe_insert(data_path="scale", frame=1)

    def trigger_impact(frame, pos_xyz):
        spark_ring.location = pos_xyz
        spark_ring.keyframe_insert(data_path="location", frame=frame - 1)
        spark_ring.scale = (0.001, 0.001, 0.001)
        spark_ring.keyframe_insert(data_path="scale", frame=frame - 1)
        spark_ring.scale = (1.6, 1.6, 1.6)
        spark_ring.keyframe_insert(data_path="scale", frame=frame + 2)
        spark_ring.scale = (0.001, 0.001, 0.001)
        spark_ring.keyframe_insert(data_path="scale", frame=frame + 7)

    # --- 4. BIÊN ĐẠO 60 GIÂY (1440 FRAMES) VÕ THUẬT 6 HỒI ---
    # Hồi 1 (0–10s | f=1..240): Tiến vào đối đầu & vào thế thủ
    apply_pose(hero, "stand", 1, (0.0, -2.4, 0.0), 0, hero_base_yaw, fist=0.4)
    apply_pose(enemy, "stand", 1, (0.0, 2.4, 0.0), 0, enemy_base_yaw, fist=0.4)
    apply_pose(hero, "guard", 120, (0.0, -1.15, 0.0), 0, hero_base_yaw, fist=1.0)
    apply_pose(enemy, "guard", 120, (0.0, 1.15, 0.0), 0, enemy_base_yaw, fist=1.0)
    apply_pose(hero, "guard", 220, (0.0, -0.95, 0.0), 8, hero_base_yaw, fist=1.0)
    apply_pose(enemy, "guard", 220, (0.0, 0.95, 0.0), -8, enemy_base_yaw, fist=1.0)

    # Hồi 2 (10–22s | f=240..528): Loạt đấm Jab-Cross tốc độ cao & đỡ phản công
    beats_h2 = [
        (252, "jab_l", (0, -0.78, 0), "block", (0, 0.85, 0), True),
        (276, "guard", (0, -0.88, 0), "guard", (0, 0.85, 0), False),
        (300, "cross_r", (0, -0.70, 0), "hit_recoil", (0, 1.08, 0), True),
        (336, "guard", (0, -0.82, 0), "guard", (0, 1.00, 0), False),
        (372, "block", (0, -0.92, 0), "jab_l", (0, 0.72, 0), True),
        (408, "block", (0, -1.02, 0), "cross_r", (0, 0.65, 0), True),
        (444, "uppercut_r", (0, -0.72, 0), "hit_recoil", (0, 1.10, 0), True),
        (492, "guard", (0, -0.95, 0), "guard", (0, 1.05, 0), False),
    ]
    for fr, hp, hl, ep, el, imp in beats_h2:
        apply_pose(hero, hp, fr, hl, 0, hero_base_yaw)
        apply_pose(enemy, ep, fr, el, 0, enemy_base_yaw)
        if imp:
            trigger_impact(fr, (0.0, 0.5 * (hl[1] + el[1]), 1.38))

    # Hồi 3 (22–34s | f=528..816): Đá tống trực diện & Đá vòng cầu xoay người (Roundhouse Kick)
    beats_h3 = [
        (552, "front_kick_r", (0, -0.75, 0), "hit_recoil", (0, 1.25, 0), True),
        (595, "guard", (0, -0.95, 0), "guard", (0, 1.25, 0), False),
        (640, "jab_l", (0, -0.70, 0), "block", (0, 0.90, 0), True),
        (676, "roundhouse_r", (0, -0.68, 0), "hit_recoil", (0, 1.40, 0), True),
        (730, "guard", (0, -0.90, 0), "guard", (0, 1.35, 0), False),
        (775, "cross_r", (0, -0.72, 0), "block", (0, 0.92, 0), True),
        (810, "guard", (0, -0.95, 0), "guard", (0, 1.00, 0), False),
    ]
    for fr, hp, hl, ep, el, imp in beats_h3:
        apply_pose(hero, hp, fr, hl, 0, hero_base_yaw)
        apply_pose(enemy, ep, fr, el, 0, enemy_base_yaw)
        if imp:
            trigger_impact(fr, (0.0, 0.5 * (hl[1] + el[1]), 1.32))

    # Hồi 4 (34–46s | f=816..1104): Mạc Khiêu phản công dồn ép Hỏa Đỏ -> Hỏa Đỏ gạt đòn phản kích
    beats_h4 = [
        (850, "block", (0, -1.10, 0), "jab_l", (0, 0.65, 0), True),
        (884, "hit_recoil", (0, -1.45, 0), "cross_r", (0, 0.45, 0), True),
        (924, "block", (0, -1.50, 0), "front_kick_r", (0, 0.40, 0), True),
        (968, "guard", (0, -1.35, 0), "guard", (0, 0.70, 0), False),
        (1012, "block", (0, -1.20, 0), "roundhouse_r", (0, 0.55, 0), True),
        (1056, "uppercut_r", (0, -0.80, 0), "hit_recoil", (0, 1.05, 0), True),
        (1096, "guard", (0, -0.95, 0), "guard", (0, 1.15, 0), False),
    ]
    for fr, hp, hl, ep, el, imp in beats_h4:
        apply_pose(hero, hp, fr, hl, 0, hero_base_yaw)
        apply_pose(enemy, ep, fr, el, 0, enemy_base_yaw)
        if imp:
            trigger_impact(fr, (0.0, 0.5 * (hl[1] + el[1]), 1.35))

    # Hồi 5 (46–55s | f=1104..1320): Liên hoàn Hỏa Quyền kết liễu (Jab -> Cross -> Uppercut -> Roundhouse -> Finisher)
    beats_h5 = [
        (1136, "jab_l", (0, -0.72, 0), "hit_recoil", (0, 0.95, 0), True),
        (1168, "cross_r", (0, -0.62, 0), "hit_recoil", (0, 1.12, 0), True),
        (1204, "uppercut_r", (0, -0.58, 0), "hit_recoil", (0, 1.28, 0), True),
        (1248, "roundhouse_r", (0, -0.65, 0), "hit_recoil", (0, 1.55, 0), True),
        (1292, "cross_r", (0, -0.45, 0), "hit_recoil", (0, 1.85, 0), True),
    ]
    for fr, hp, hl, ep, el, imp in beats_h5:
        apply_pose(hero, hp, fr, hl, 0, hero_base_yaw)
        apply_pose(enemy, ep, fr, el, 0, enemy_base_yaw)
        if imp:
            trigger_impact(fr, (0.0, 0.5 * (hl[1] + el[1]), 1.40))

    # Hồi 6 (55–60s | f=1320..1440): Mạc Khiêu khuỵu gối thua cuộc, Hỏa Đỏ đứng hiên ngang thắng lợi
    apply_pose(hero, "guard", 1348, (0.0, -0.65, 0.0), 0, hero_base_yaw, fist=1.0)
    apply_pose(enemy, "kneel_defeat", 1356, (0.0, 1.95, -0.42), 0, enemy_base_yaw, fist=0.5)
    apply_pose(hero, "stand", 1440, (0.0, -0.65, 0.0), 12, hero_base_yaw, fist=0.8)
    apply_pose(enemy, "kneel_defeat", 1440, (0.0, 1.95, -0.42), 0, enemy_base_yaw, fist=0.5)

    # --- 5. ÁNH SÁNG ĐIỆN ẢNH 3 ĐIỂM + RIM LIGHT ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.025, 0.04, 1.0)
        bg.inputs[1].default_value = 0.75

    def add_light(name, loc, rot_deg, energy, color, size=2.5):
        bpy.ops.object.light_add(type="AREA", location=loc, rotation=tuple(math.radians(a) for a in rot_deg))
        l = bpy.context.active_object
        l.name = name
        l.data.energy = energy
        l.data.color = color
        l.data.size = size

    add_light("Key_Main", (3.5, -2.5, 3.6), (55, 0, 52), 340.0, (1.0, 0.94, 0.88), 3.0)
    add_light("Fill_Cool", (-3.5, 1.5, 2.8), (64, 0, -115), 180.0, (0.72, 0.84, 1.0), 3.2)
    add_light("Rim_Red", (-2.2, -4.5, 2.6), (68, 0, -25), 420.0, (1.0, 0.22, 0.10), 2.2)
    add_light("Rim_Blue", (2.2, 4.5, 2.6), (-68, 0, 25), 380.0, (0.12, 0.72, 1.0), 2.2)

    # --- 6. 8 GÓC MÁY HÀNH ĐỘNG ĐIỆN ẢNH CẮT CẢNH LIÊN TỤC TRONG 60 GIÂY ---
    mid_tgt = bpy.data.objects.new("FightCenter", None)
    scene.collection.objects.link(mid_tgt)
    mid_tgt.location = (0.0, 0.1, 1.25)

    hero_head_tgt = bpy.data.objects.new("HeroHeadTarget", None)
    scene.collection.objects.link(hero_head_tgt)
    hero_head_tgt.parent = hero["root"]
    hero_head_tgt.location = (0.0, 0.0, 1.50)

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
        (1, make_cam("C1_WideIntro", (3.6, -1.2, 1.35), (3.1, 0.4, 1.45), 1, 144, mid_tgt, 34.0)),
        (145, make_cam("C2_HeroClose", (1.1, 0.6, 1.45), (0.85, 0.2, 1.52), 145, 240, hero_head_tgt, 52.0)),
        (241, make_cam("C3_SideActionR", (3.0, 0.0, 1.25), (2.7, 0.6, 1.35), 241, 420, mid_tgt, 38.0)),
        (421, make_cam("C4_LowKickL", (-2.8, -0.4, 0.72), (-2.6, 0.9, 0.95), 421, 660, mid_tgt, 35.0)),
        (661, make_cam("C5_OverShoulder", (-1.5, -2.5, 1.55), (-1.1, -1.9, 1.48), 661, 840, mid_tgt, 40.0)),
        (841, make_cam("C6_CounterR", (2.9, -1.0, 1.15), (2.6, -0.2, 1.35), 841, 1080, mid_tgt, 36.0)),
        (1081, make_cam("C7_FinisherDash", (-2.4, 0.8, 0.85), (2.2, 1.1, 1.15), 1081, 1320, mid_tgt, 38.0)),
        (1321, make_cam("C8_VictoryHero", (1.5, 0.9, 1.05), (-1.2, 0.8, 1.42), 1321, 1440, hero_head_tgt, 45.0)),
    ]
    scene.camera = cam_schedule[0][1]
    for fstart, cam_obj in cam_schedule:
        mk = scene.timeline_markers.new(cam_obj.name, frame=fstart)
        mk.camera = cam_obj

    # --- 7. RENDER CYCLES GPU TỐI ƯU TỐC ĐỘ CHO 1 PHÚT (1440 FRAMES @ 720p) ---
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    gpu_names = []
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
            if dev.type != "CPU":
                gpu_names.append(dev.name)
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"
    print(f"GPU đang dùng: {', '.join(gpu_names) if gpu_names else 'CPU'} | Cycles device={scene.cycles.device}", flush=True)

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = out_raw_mp4
    os.makedirs(os.path.dirname(out_raw_mp4), exist_ok=True)
    print(f"Đang render 1 phút võ thuật 3D ({TOTAL_FRAMES} frames @ 1280x720 Cycles GPU)...", flush=True)
    bpy.ops.render.render(animation=True)
    print("-> Render Blender hoàn tất:", out_raw_mp4, flush=True)


def download_file(urls, dest_path, label):
    if isinstance(urls, str):
        urls = [urls]
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 300_000:
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"Đang tải {label} từ {url}...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            if os.path.getsize(dest_path) > 300_000:
                print(f"-> Xong {label} ({os.path.getsize(dest_path)/1024/1024:.2f} MB)", flush=True)
                return
        except Exception as e:
            last_err = e
            print(f"  Bỏ qua link lỗi ({e}), thử link dự phòng...", flush=True)
    raise RuntimeError(f"Không tải được {label}: {last_err}")


def generate_impact_sfx_track(out_wav, duration_sec=60.0):
    """Tạo track âm thanh tiếng đấm/đá (Impact Punch/Kick SFX) khớp chính xác từng frame trúng đòn."""
    sr = 44100
    n_samples = int(sr * duration_sec)
    buf = [0.0] * n_samples
    hit_frames = [252, 300, 372, 408, 444, 552, 640, 676, 775, 850, 884, 924, 1012, 1056, 1136, 1168, 1204, 1248, 1292]
    import random
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
        frames_bytes = b"".join(struct.pack("<h", int(s * 32000)) for s in buf)
        wf.writeframes(frames_bytes)


def synth_dialogue_lines(tmp_dir):
    """Tạo các câu thoại tiếng Việt cho Hỏa Đỏ và Mạc Khiêu đúng nhịp trận đấu 60 giây."""
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
        run_blender_martial_arts(sys.argv[idx + 1], sys.argv[idx + 2], sys.argv[idx + 3])
        return

    models_dir = "/content/Test/assets/models_3d"
    out_dir = "/content/Test/build_output"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    hero_glb = os.path.join(models_dir, "human_hero_male.glb")
    enemy_glb = os.path.join(models_dir, "human_enemy_soldier.glb")
    raw_mp4 = os.path.join(out_dir, "demo_martial_arts_1min_raw.mp4")
    sfx_wav = os.path.join(out_dir, "demo_martial_arts_sfx.wav")
    final_mp4 = os.path.join(out_dir, "demo_sentai_3d_scene.mp4")

    download_file([
        "https://raw.githubusercontent.com/readyplayerme/visage/main/public/male.glb",
        "https://raw.githubusercontent.com/readyplayerme/visage/master/public/male.glb",
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb",
    ], hero_glb, "Nhân vật người thật Hỏa Đỏ")
    download_file([
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb",
    ], enemy_glb, "Nhân vật người thật Đối thủ (Mạc Khiêu)")

    cmd = [
        "/content/blender/blender", "--background",
        "--python", os.path.abspath(__file__), "--",
        "--blender-render", hero_glb, enemy_glb, raw_mp4,
    ]
    subprocess.run(cmd, check=True)

    generate_impact_sfx_track(sfx_wav, duration_sec=60.0)
    dialogue_items = synth_dialogue_lines(out_dir)
    mux_full_audio(raw_mp4, sfx_wav, dialogue_items, final_mp4)
    print(f"HOÀN TẤT VIDEO VÕ THUẬT 1 PHÚT: {final_mp4} ({os.path.getsize(final_mp4)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
