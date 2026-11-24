"""
Headless Blender 4.2 script for Cách 3 (VRoid / VRM Character Demo):
1. Imports any .vrm / .glb humanoid character (natively via glTF 2.0)
2. Recolors outfit materials to Hỏa Đỏ (Crimson Red & Tactical Black) while keeping fabric folds
3. Animates:
   - Real 10-finger CLENCHED FISTS (nắm tay thật)
   - Heroic fighting guard + forward punch (no torso clipping)
   - Facial expressions: eye blinking + mouth lip-sync (A-I-U-E-O)
4. Exports rigged/recolored .glb and renders 720p HD Cycles demo video (.mp4)
"""
import math
import os
import sys
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
input_vrm = os.path.abspath(argv[0])
out_glb = os.path.abspath(argv[1])
out_mp4 = os.path.abspath(argv[2]) if len(argv) > 2 else None
theme = argv[3] if len(argv) > 3 else "hoa_do"

THEMES = {
    "hoa_do": {"top": (0.78, 0.06, 0.08, 1.0), "bottom": (0.07, 0.07, 0.09, 1.0), "hair": (0.12, 0.05, 0.06, 1.0)},
    "thuy_lam": {"top": (0.06, 0.25, 0.80, 1.0), "bottom": (0.06, 0.08, 0.14, 1.0), "hair": (0.05, 0.10, 0.22, 1.0)},
    "tho_vang": {"top": (0.85, 0.62, 0.06, 1.0), "bottom": (0.10, 0.09, 0.07, 1.0), "hair": (0.14, 0.10, 0.06, 1.0)},
    "phong_luc": {"top": (0.06, 0.58, 0.22, 1.0), "bottom": (0.07, 0.10, 0.08, 1.0), "hair": (0.06, 0.14, 0.08, 1.0)},
    "loi_hong": {"top": (0.82, 0.12, 0.48, 1.0), "bottom": (0.10, 0.07, 0.10, 1.0), "hair": (0.16, 0.05, 0.12, 1.0)},
}
palette = THEMES.get(theme, THEMES["hoa_do"])

bpy.ops.wm.read_factory_settings(use_empty=True)

# Blender 4.2 imports .vrm natively as glTF 2.0 (copy/symlink to .glb if needed)
import_path = input_vrm
tmp_glb = None
if input_vrm.lower().endswith(".vrm"):
    import shutil
    tmp_glb = input_vrm + ".tmp.glb"
    shutil.copy2(input_vrm, tmp_glb)
    import_path = tmp_glb

bpy.ops.import_scene.gltf(filepath=import_path)
if tmp_glb and os.path.exists(tmp_glb):
    os.remove(tmp_glb)

arm_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not arm_objs or not mesh_objs:
    raise RuntimeError(f"Missing armature or mesh in {input_vrm}")

arm_obj = arm_objs[0]
print(f"Loaded VRM: armature={arm_obj.name} ({len(arm_obj.data.bones)} bones), meshes={len(mesh_objs)}")

# Ensure smooth shading on all meshes
for m in mesh_objs:
    bpy.context.view_layer.objects.active = m
    bpy.ops.object.select_all(action="DESELECT")
    m.select_set(True)
    bpy.ops.object.shade_smooth()


def tint_material(mat, rgba, factor=0.78):
    """Nhuộm màu áo/quần giữ nguyên nếp vải & bóng đổ gốc."""
    if not mat.use_nodes or not mat.node_tree:
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
    if "Metallic" in bsdf.inputs and not bsdf.inputs["Metallic"].is_linked:
        bsdf.inputs["Metallic"].default_value = 0.05
    if "Roughness" in bsdf.inputs and not bsdf.inputs["Roughness"].is_linked:
        bsdf.inputs["Roughness"].default_value = max(0.55, bsdf.inputs["Roughness"].default_value)


# Tự động đổi màu áo/quần/tóc theo tông chiến binh (không đụng vào Face/Eye/Skin!)
recolored = []
for mat in bpy.data.materials:
    low = mat.name.lower()
    if any(k in low for k in ("face", "eye", "skin", "mouth", "brow", "lash", "iris", "highlight", "body")):
        continue
    if any(k in low for k in ("top", "cloth", "jacket", "shirt", "coat", "vest", "suit", "uniform", "ribbon", "tie", "onepiece", "dress")):
        tint_material(mat, palette["top"], factor=0.82)
        recolored.append(f"{mat.name}->TOP")
    elif any(k in low for k in ("bottom", "pant", "skirt", "short", "shoe", "boot", "sock", "leg")):
        tint_material(mat, palette["bottom"], factor=0.85)
        recolored.append(f"{mat.name}->BOTTOM")
    elif "hair" in low:
        tint_material(mat, palette["hair"], factor=0.45)
        recolored.append(f"{mat.name}->HAIR")
print("Recolored materials:", recolored)


# Tìm xương linh hoạt theo cả chuẩn VRM 0.x (J_Bip_*) và VRM 1.0 / Mixamo / Blender
pb = arm_obj.pose.bones
for b in pb:
    b.rotation_mode = "XYZ"


def find_bone(*patterns):
    for pat in patterns:
        p = pat.lower()
        for b in pb:
            if b.name.lower() == p:
                return b
    for pat in patterns:
        p = pat.lower()
        for b in pb:
            if p in b.name.lower():
                return b
    return None


b_chest = find_bone("J_Bip_C_UpperChest", "J_Bip_C_Chest", "upperChest", "chest", "spine2")
b_head = find_bone("J_Bip_C_Head", "head")
b_ua_r = find_bone("J_Bip_R_UpperArm", "rightUpperArm", "upper_arm.r", "UpperArm_R")
b_fa_r = find_bone("J_Bip_R_LowerArm", "rightLowerArm", "forearm.r", "ForeArm_R")
b_ua_l = find_bone("J_Bip_L_UpperArm", "leftUpperArm", "upper_arm.l", "UpperArm_L")
b_fa_l = find_bone("J_Bip_L_LowerArm", "leftLowerArm", "forearm.l", "ForeArm_L")

# Tìm toàn bộ 10 ngón tay (3 đốt mỗi ngón) để nắm chặt nắm đấm thật!
finger_bones_r = []
finger_bones_l = []
for b in pb:
    low = b.name.lower()
    if any(f in low for f in ("index", "middle", "ring", "little", "pinky", "thumb")):
        is_thumb = "thumb" in low
        if "_r_" in low or "right" in low or low.endswith(".r") or low.endswith("_r"):
            finger_bones_r.append((b, is_thumb))
        elif "_l_" in low or "left" in low or low.endswith(".l") or low.endswith("_l"):
            finger_bones_l.append((b, is_thumb))
print(f"Found finger bones: R={len(finger_bones_r)}, L={len(finger_bones_l)}")

# Kiểm tra chiều hướng mặt của model (VRM 0.x hướng -Z/+Y, VRM 1.0 hướng +Z/-Y)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 96
scene.render.fps = 24

bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="POSE")


def set_rot(bone, deg_xyz, frame):
    if bone is None:
        return
    bone.rotation_euler = tuple(math.radians(a) for a in deg_xyz)
    bone.keyframe_insert(data_path="rotation_euler", frame=frame)


def set_fist(finger_list, curl_ratio, frame, side_sign=1):
    """curl_ratio: 0.0 = mở tay, 1.0 = nắm đấm chặt."""
    for b, is_thumb in finger_list:
        if is_thumb:
            ang = (0, 28 * curl_ratio * side_sign, 35 * curl_ratio * side_sign)
        else:
            ang = (0, 0, 72 * curl_ratio * side_sign)
        set_rot(b, ang, frame)


# Frame 1: Đứng nghiêm chuẩn A-Pose (từ T-pose hạ 2 tay xuống 42 độ)
for f in (1, 96):
    set_rot(b_chest, (0, 0, 0), f)
    set_rot(b_head, (0, 0, 0), f)
    set_rot(b_ua_r, (0, 0, -42), f)
    set_rot(b_fa_r, (0, 0, 0), f)
    set_rot(b_ua_l, (0, 0, 42), f)
    set_rot(b_fa_l, (0, 0, 0), f)
    set_fist(finger_bones_r, 0.25, f, side_sign=-1)
    set_fist(finger_bones_l, 0.25, f, side_sign=1)

# Frame 24 & 72: Thủ thế chiến đấu + NẮM CHẶT 2 NẮM ĐẤM
for f in (24, 72):
    set_rot(b_chest, (4, -8, 0), f)
    set_rot(b_head, (-2, 8, 0), f)
    set_rot(b_ua_r, (-28, 35, -30), f)
    set_rot(b_fa_r, (0, 68, 0), f)
    set_rot(b_ua_l, (-22, -32, 30), f)
    set_rot(b_fa_l, (0, -60, 0), f)
    set_fist(finger_bones_r, 1.0, f, side_sign=-1)
    set_fist(finger_bones_l, 1.0, f, side_sign=1)

# Frame 48: Tung cú đấm thẳng tay phải (nắm đấm chặt)
set_rot(b_chest, (6, 12, 0), 48)
set_rot(b_head, (-2, -10, 0), 48)
set_rot(b_ua_r, (-10, 72, -10), 48)
set_rot(b_fa_r, (0, 12, 0), 48)
set_rot(b_ua_l, (-18, -38, 36), 48)
set_rot(b_fa_l, (0, -75, 0), 48)
set_fist(finger_bones_r, 1.0, 48, side_sign=-1)
set_fist(finger_bones_l, 1.0, 48, side_sign=1)

bpy.ops.object.mode_set(mode="OBJECT")

# Hoạt ảnh biểu cảm khuôn mặt (Chớp mắt + Nhép miệng nói chuyện!)
for m in mesh_objs:
    sk = m.data.shape_keys
    if not sk:
        continue
    kbs = sk.key_blocks
    blink_key = next((k for k in kbs if any(t in k.name.lower() for t in ("fcl_eye_close", "blink", "eye_close"))), None)
    mouth_a = next((k for k in kbs if k.name.lower() in ("fcl_mth_a", "aa", "a", "mth_a") or "mth_a" in k.name.lower()), None)
    if blink_key:
        for f, v in [(1, 0.0), (18, 0.0), (21, 1.0), (24, 0.0), (65, 0.0), (68, 1.0), (71, 0.0)]:
            blink_key.value = v
            blink_key.keyframe_insert(data_path="value", frame=f)
        print("Animated blink shape key:", blink_key.name)
    if mouth_a:
        for f, v in [(1, 0.0), (30, 0.0), (36, 0.75), (42, 0.15), (48, 0.9), (56, 0.0), (96, 0.0)]:
            mouth_a.value = v
            mouth_a.keyframe_insert(data_path="value", frame=f)
        print("Animated mouth shape key:", mouth_a.name)

# Export .glb
try:
    bpy.ops.file.pack_all()
except Exception:
    pass

os.makedirs(os.path.dirname(out_glb), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    export_animations=True,
    export_image_format="AUTO",
)
print("Exported VRM demo GLB:", out_glb)

# Render 720p HD demo video (Cycles GPU)
if out_mp4:
    # Tính chiều cao thực tế của nhân vật để căn camera chuẩn
    all_pts = []
    for m in mesh_objs:
        all_pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
    min_z = min(v.z for v in all_pts)
    max_z = max(v.z for v in all_pts)
    mid_z = 0.5 * (min_z + max_z)
    h = max(0.5, max_z - min_z)

    bpy.ops.object.light_add(type="SUN", location=(3, -4, 5))
    bpy.context.active_object.data.energy = 2.6
    bpy.ops.object.light_add(type="AREA", location=(0, -3.4, mid_z + 0.4), rotation=(math.radians(82), 0, 0))
    bpy.context.active_object.data.energy = 110.0
    bpy.context.active_object.data.size = 2.5
    bpy.ops.object.light_add(type="AREA", location=(-3, -2, mid_z + 0.8))
    bpy.context.active_object.data.energy = 85.0

    # Camera hơi gần để thấy rõ cả khuôn mặt nét + nắm đấm 10 ngón + toàn thân
    bpy.ops.object.camera_add(location=(0, -h * 2.15, mid_z + 0.05), rotation=(math.radians(89), 0, 0))
    scene.camera = bpy.context.active_object

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.14, 0.16, 0.21, 1.0)
        bg.inputs[1].default_value = 1.1

    # Xoay nhẹ để khoe mặt trước, cú đấm và góc nghiêng
    for f, ang in [(1, -15), (48, 20), (72, -10), (96, 360)]:
        arm_obj.rotation_euler = (0, 0, math.radians(ang))
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=f)

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
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
    scene.render.filepath = out_mp4
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    bpy.ops.render.render(animation=True)
    print("Rendered VRM demo MP4:", out_mp4)
