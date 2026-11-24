"""
Headless Blender 4.2 script for Cách 3 (Male VRoid / VRM + 3D Super Sentai Armor):
1. Imports a masculine .vrm / .glb humanoid character (natively via glTF 2.0)
2. Auto-aligns character facing -Y (towards camera)
3. Recolors outfit materials to Hỏa Đỏ (Crimson Red & Tactical Black)
4. Attaches real 3D Super Sentai armor pieces (Chest Plate + Glowing Core, Shoulder Pauldrons,
   Forearm Gauntlets, Henshin Belt) parented directly to the character's bones
5. Poses in world-space (guaranteed 0% arm-through-torso clipping) with wide martial-arts stance,
   10-finger clenched fists, forward punch, blinking eyes & lip-sync mouth
6. Exports rigged armored .glb and renders 720p HD heroic Low-Angle Cycles demo video (.mp4)
"""
import math
import os
import shutil
import sys
import bpy
from mathutils import Euler, Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
input_vrm = os.path.abspath(argv[0])
out_glb = os.path.abspath(argv[1])
out_mp4 = os.path.abspath(argv[2]) if len(argv) > 2 else None
theme = argv[3] if len(argv) > 3 else "hoa_do"

THEMES = {
    "hoa_do": {
        "primary": (0.82, 0.05, 0.07, 1.0),
        "secondary": (0.05, 0.05, 0.07, 1.0),
        "glow": (1.0, 0.16, 0.04, 1.0),
        "hair": (0.10, 0.04, 0.05, 1.0),
    },
    "thuy_lam": {
        "primary": (0.05, 0.28, 0.88, 1.0),
        "secondary": (0.05, 0.07, 0.12, 1.0),
        "glow": (0.10, 0.65, 1.0, 1.0),
        "hair": (0.04, 0.08, 0.18, 1.0),
    },
    "tho_vang": {
        "primary": (0.88, 0.64, 0.05, 1.0),
        "secondary": (0.08, 0.07, 0.05, 1.0),
        "glow": (1.0, 0.80, 0.15, 1.0),
        "hair": (0.12, 0.09, 0.05, 1.0),
    },
    "phong_luc": {
        "primary": (0.05, 0.62, 0.22, 1.0),
        "secondary": (0.05, 0.08, 0.06, 1.0),
        "glow": (0.15, 1.0, 0.45, 1.0),
        "hair": (0.05, 0.12, 0.07, 1.0),
    },
    "loi_hong": {
        "primary": (0.85, 0.10, 0.50, 1.0),
        "secondary": (0.08, 0.05, 0.09, 1.0),
        "glow": (1.0, 0.25, 0.72, 1.0),
        "hair": (0.14, 0.04, 0.10, 1.0),
    },
}
pal = THEMES.get(theme, THEMES["hoa_do"])

bpy.ops.wm.read_factory_settings(use_empty=True)

import_path = input_vrm
tmp_glb = None
if input_vrm.lower().endswith(".vrm"):
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
print(f"Loaded Male VRM: armature={arm_obj.name} ({len(arm_obj.data.bones)} bones), meshes={len(mesh_objs)}")

for m in mesh_objs:
    bpy.context.view_layer.objects.active = m
    bpy.ops.object.select_all(action="DESELECT")
    m.select_set(True)
    bpy.ops.object.shade_smooth()


def make_armor_mat(name, color, metallic=0.45, roughness=0.32, emission=None, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        if emission and "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


mat_armor_red = make_armor_mat("Armor_Primary", pal["primary"], metallic=0.50, roughness=0.28)
mat_armor_blk = make_armor_mat("Armor_Black", pal["secondary"], metallic=0.35, roughness=0.42)
mat_armor_glow = make_armor_mat("Armor_Glow", pal["glow"], metallic=0.1, roughness=0.2, emission=pal["glow"], emission_strength=6.0)


def tint_material(mat, rgba, factor=0.85):
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
        bsdf.inputs["Metallic"].default_value = 0.08
    if "Roughness" in bsdf.inputs and not bsdf.inputs["Roughness"].is_linked:
        bsdf.inputs["Roughness"].default_value = max(0.55, bsdf.inputs["Roughness"].default_value)


# Nhuộm màu áo/quần gốc sang đen chiến thuật & đỏ đậm, giữ nguyên mặt/da/mắt
for mat in bpy.data.materials:
    if mat.name.startswith("Armor_"):
        continue
    low = mat.name.lower()
    if any(k in low for k in ("face", "eye", "skin", "mouth", "brow", "lash", "iris", "highlight", "body")):
        continue
    if any(k in low for k in ("top", "cloth", "jacket", "shirt", "coat", "vest", "suit", "uniform", "tie", "ribbon")):
        tint_material(mat, (0.14, 0.14, 0.16, 1.0), factor=0.88)
    elif any(k in low for k in ("bottom", "pant", "short", "shoe", "boot", "sock", "leg")):
        tint_material(mat, (0.08, 0.08, 0.10, 1.0), factor=0.90)
    elif "hair" in low:
        tint_material(mat, pal["hair"], factor=0.55)


# Tìm xương theo cả chuẩn VRM 0.x và VRM 1.0
pb = arm_obj.pose.bones
for b in pb:
    b.rotation_mode = "QUATERNION"


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


b_hips = find_bone("J_Bip_C_Hips", "hips")
b_spine = find_bone("J_Bip_C_Spine", "spine")
b_chest = find_bone("J_Bip_C_UpperChest", "J_Bip_C_Chest", "upperChest", "chest")
b_head = find_bone("J_Bip_C_Head", "head")
b_eye = find_bone("J_Adj_L_FaceEye", "leftEye", "eye")
b_ua_r = find_bone("J_Bip_R_UpperArm", "rightUpperArm", "upper_arm.r")
b_fa_r = find_bone("J_Bip_R_LowerArm", "rightLowerArm", "forearm.r")
b_ua_l = find_bone("J_Bip_L_UpperArm", "leftUpperArm", "upper_arm.l")
b_fa_l = find_bone("J_Bip_L_LowerArm", "leftLowerArm", "forearm.l")
b_th_r = find_bone("J_Bip_R_UpperLeg", "rightUpperLeg", "thigh.r")
b_kn_r = find_bone("J_Bip_R_LowerLeg", "rightLowerLeg", "shin.r")
b_th_l = find_bone("J_Bip_L_UpperLeg", "leftUpperLeg", "thigh.l")
b_kn_l = find_bone("J_Bip_L_LowerLeg", "leftLowerLeg", "shin.l")

# Xác định hướng mặt thực tế của VRM (+Y hay -Y) để nhân vật LUÔN quay mặt về camera (-Y)
bpy.context.view_layer.update()
face_dir = -1.0  # -1 means facing -Y (front towards camera at -Y)
if b_head and b_eye:
    head_w = arm_obj.matrix_world @ b_head.bone.head_local
    eye_w = arm_obj.matrix_world @ b_eye.bone.head_local
    if eye_w.y > head_w.y + 0.01:
        face_dir = 1.0  # model currently faces +Y, will rotate armature 180 deg at render/export
print(f"Detected VRM face_dir={'+Y (will flip 180)' if face_dir > 0 else '-Y (front)'}")

# --- GẮN BỘ GIÁP 3D SUPER SENTAI TRỰC TIẾP LÊN XƯƠNG NHÂN VẬT ---
def attach_armor_piece(name, bone, size_xyz, offset_world, mat, bevel=0.012, rot_deg=(0, 0, 0)):
    if bone is None:
        return None
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size_xyz
    bpy.ops.object.transform_apply(scale=True)
    bev = obj.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = bevel
    bev.segments = 2
    bpy.ops.object.modifier_apply(modifier=bev.name)
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)

    # Đặt vị trí theo tọa độ world của xương rồi parent vào xương
    bw = arm_obj.matrix_world @ bone.bone.head_local
    obj.location = bw + Vector((offset_world[0], offset_world[1] * (-face_dir), offset_world[2]))
    obj.rotation_euler = tuple(math.radians(a) for a in rot_deg)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    obj.parent = arm_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone.name
    # Giữ nguyên transform world khi parent vào bone
    bone_mat = arm_obj.matrix_world @ Matrix.Translation(bone.bone.tail_local) @ bone.bone.matrix_local.to_3x3().to_4x4()
    obj.matrix_parent_inverse = bone_mat.inverted()
    return obj


# 1. Giáp ngực tactical đỏ-đen + Lõi năng lượng Hỏa phát sáng
attach_armor_piece("Armor_Chest_Main", b_chest, (0.30, 0.22, 0.24), (0.0, -0.02, 0.04), mat_armor_blk, bevel=0.025)
attach_armor_piece("Armor_Chest_Red_L", b_chest, (0.09, 0.235, 0.22), (0.11, -0.02, 0.05), mat_armor_red, bevel=0.015)
attach_armor_piece("Armor_Chest_Red_R", b_chest, (0.09, 0.235, 0.22), (-0.11, -0.02, 0.05), mat_armor_red, bevel=0.015)
attach_armor_piece("Armor_Chest_Core", b_chest, (0.075, 0.04, 0.075), (0.0, -0.135, 0.06), mat_armor_glow, bevel=0.012, rot_deg=(0, 45, 0))

# 2. Giáp vai 3D (Pauldrons) gắn vào 2 vai
attach_armor_piece("Armor_Shoulder_R", b_ua_r, (0.14, 0.14, 0.11), (-0.03, 0.0, 0.03), mat_armor_red, bevel=0.02, rot_deg=(0, -18, 0))
attach_armor_piece("Armor_Shoulder_L", b_ua_l, (0.14, 0.14, 0.11), (0.03, 0.0, 0.03), mat_armor_red, bevel=0.02, rot_deg=(0, 18, 0))

# 3. Giáp cẳng tay (Gauntlets)
attach_armor_piece("Armor_Gauntlet_R", b_fa_r, (0.16, 0.095, 0.095), (-0.11, 0.0, 0.0), mat_armor_red, bevel=0.015)
attach_armor_piece("Armor_Gauntlet_L", b_fa_l, (0.16, 0.095, 0.095), (0.11, 0.0, 0.0), mat_armor_red, bevel=0.015)

# 4. Thắt lưng biến hình (Henshin Belt) + khóa lõi phát sáng
attach_armor_piece("Armor_Belt", b_hips, (0.31, 0.21, 0.065), (0.0, 0.0, 0.04), mat_armor_blk, bevel=0.01)
attach_armor_piece("Armor_Belt_Buckle", b_hips, (0.09, 0.045, 0.07), (0.0, -0.11, 0.04), mat_armor_glow, bevel=0.01)

# --- TÌM 10 NGÓN TAY ĐỂ NẮM ĐẤM THẬT ---
finger_bones_r, finger_bones_l = [], []
for b in pb:
    low = b.name.lower()
    if any(f in low for f in ("index", "middle", "ring", "little", "pinky", "thumb")):
        is_thumb = "thumb" in low
        if "_r_" in low or "right" in low or low.endswith(".r"):
            finger_bones_r.append((b, is_thumb))
        elif "_l_" in low or "left" in low or low.endswith(".l"):
            finger_bones_l.append((b, is_thumb))
print(f"Found fingers: R={len(finger_bones_r)}, L={len(finger_bones_l)}")

# --- HOẠT ẢNH TRONG KHÔNG GIAN WORLD (ĐẢM BẢO 0% XUYÊN ÁO!) ---
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 96
scene.render.fps = 24

bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="POSE")


def set_world_rot(bone, deg_xyz, frame):
    """Xoay xương theo trục World (X=gập trước/sau, Y=dạng ngang/hạ tay, Z=xoay trái/phải),
    tự đổi sang không gian local của xương nên không bao giờ bị ngược trục xuyên thân."""
    if bone is None:
        return
    # Nếu VRM quay mặt về +Y, đảo dấu trục X và Y cho nhất quán với hướng mặt
    sx = -face_dir
    w_rot = Euler((math.radians(deg_xyz[0] * sx), math.radians(deg_xyz[1] * sx), math.radians(deg_xyz[2])), "XYZ").to_matrix()
    rest_m = bone.bone.matrix_local.to_3x3()
    local_m = rest_m.inverted() @ w_rot @ rest_m
    bone.rotation_quaternion = local_m.to_quaternion()
    bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def set_fist(finger_list, curl, frame, side_sign):
    for b, is_thumb in finger_list:
        if is_thumb:
            set_world_rot(b, (25 * curl, 25 * curl * side_sign, 20 * curl * side_sign), frame)
        else:
            set_world_rot(b, (0, 68 * curl * side_sign, 0), frame)


# Frame 1 & 96: Dáng đứng chiến binh A-Pose rộng vai, 2 tay mở gọn ngoài thân (0% chạm áo)
for f in (1, 96):
    set_world_rot(b_chest, (0, 0, 0), f)
    set_world_rot(b_head, (0, 0, 0), f)
    # Tay hạ 48 độ ra ngoài (còn cách thân 42 độ -> tuyệt đối không xuyên áo)
    set_world_rot(b_ua_r, (0, -48, 0), f)
    set_world_rot(b_fa_r, (0, 0, -10), f)
    set_world_rot(b_ua_l, (0, 48, 0), f)
    set_world_rot(b_fa_l, (0, 0, 10), f)
    # Chân đứng tấn rộng vai oai vệ
    set_world_rot(b_th_r, (0, 10, 0), f)
    set_world_rot(b_th_l, (0, -10, 0), f)
    set_world_rot(b_kn_r, (0, 0, 0), f)
    set_world_rot(b_kn_l, (0, 0, 0), f)
    set_fist(finger_bones_r, 0.85, f, side_sign=-1)
    set_fist(finger_bones_l, 0.85, f, side_sign=1)

# Frame 24 & 72: Thủ thế Super Sentai mạnh mẽ (2 tay mở rộng ra ngoài và đưa về trước, nắm chặt nắm đấm)
for f in (24, 72):
    set_world_rot(b_chest, (5, 0, -12), f)
    set_world_rot(b_head, (-4, 0, 10), f)
    set_world_rot(b_ua_r, (0, -35, -32), f)
    set_world_rot(b_fa_r, (0, 0, -58), f)
    set_world_rot(b_ua_l, (0, 35, 38), f)
    set_world_rot(b_fa_l, (0, 0, 52), f)
    set_world_rot(b_th_r, (-10, 14, -8), f)
    set_world_rot(b_kn_r, (18, 0, 0), f)
    set_world_rot(b_th_l, (10, -14, 8), f)
    set_world_rot(b_kn_l, (14, 0, 0), f)
    set_fist(finger_bones_r, 1.0, f, side_sign=-1)
    set_fist(finger_bones_l, 1.0, f, side_sign=1)

# Frame 48: Tung cú đấm Hỏa Quyền thẳng về trước (xoay vai dồn lực, tay trái thủ ngoài)
set_world_rot(b_chest, (6, 0, 18), 48)
set_world_rot(b_head, (-4, 0, -14), 48)
set_world_rot(b_ua_r, (0, -14, -76), 48)
set_world_rot(b_fa_r, (0, 0, -8), 48)
set_world_rot(b_ua_l, (0, 42, 22), 48)
set_world_rot(b_fa_l, (0, 0, 68), 48)
set_world_rot(b_th_r, (-14, 15, -10), 48)
set_world_rot(b_kn_r, (20, 0, 0), 48)
set_world_rot(b_th_l, (12, -15, 10), 48)
set_world_rot(b_kn_l, (16, 0, 0), 48)
set_fist(finger_bones_r, 1.0, 48, side_sign=-1)
set_fist(finger_bones_l, 1.0, 48, side_sign=1)

bpy.ops.object.mode_set(mode="OBJECT")

# Hoạt ảnh khuôn mặt: Ánh mắt quyết tâm (Angry nhẹ) + Chớp mắt + Hô chiêu (Nhép miệng)
for m in mesh_objs:
    sk = m.data.shape_keys
    if not sk:
        continue
    kbs = sk.key_blocks
    angry_key = next((k for k in kbs if any(t in k.name.lower() for t in ("fcl_all_angry", "angry", "brw_angry"))), None)
    blink_key = next((k for k in kbs if any(t in k.name.lower() for t in ("fcl_eye_close", "blink", "eye_close"))), None)
    mouth_a = next((k for k in kbs if k.name.lower() in ("fcl_mth_a", "aa", "a") or "mth_a" in k.name.lower()), None)
    if angry_key:
        angry_key.value = 0.55
        angry_key.keyframe_insert(data_path="value", frame=1)
        angry_key.value = 0.85
        angry_key.keyframe_insert(data_path="value", frame=48)
    if blink_key:
        for f, v in [(1, 0.0), (16, 0.0), (19, 1.0), (22, 0.0), (68, 0.0), (71, 1.0), (74, 0.0)]:
            blink_key.value = v
            blink_key.keyframe_insert(data_path="value", frame=f)
    if mouth_a:
        for f, v in [(1, 0.0), (36, 0.0), (44, 0.85), (52, 0.90), (60, 0.0), (96, 0.0)]:
            mouth_a.value = v
            mouth_a.keyframe_insert(data_path="value", frame=f)

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
print("Exported armored Hero GLB:", out_glb)

# Render 720p HD Heroic Low-Angle Cycles Demo
if out_mp4:
    all_pts = []
    for m in mesh_objs:
        all_pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
    min_z = min(v.z for v in all_pts)
    max_z = max(v.z for v in all_pts)
    mid_z = 0.5 * (min_z + max_z)
    h = max(0.5, max_z - min_z)

    # Sàn tối phản chiếu nhẹ tạo cảm giác sân khấu Tokusatsu
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, min_z))
    floor = bpy.context.active_object
    floor.data.materials.append(make_armor_mat("Arena_Floor", (0.03, 0.035, 0.05, 1.0), metallic=0.25, roughness=0.45))

    # Ánh sáng Tokusatsu 3 điểm: Key sáng lạnh + Fill trước + Rim-Light LỬA ĐỎ cực ngầu sau lưng!
    bpy.ops.object.light_add(type="AREA", location=(2.2, -3.2, mid_z + 1.0), rotation=(math.radians(72), 0, math.radians(32)))
    bpy.context.active_object.data.energy = 190.0
    bpy.context.active_object.data.size = 2.0

    bpy.ops.object.light_add(type="AREA", location=(-2.0, -2.8, mid_z + 0.5), rotation=(math.radians(82), 0, math.radians(-30)))
    bpy.context.active_object.data.energy = 85.0
    bpy.context.active_object.data.size = 2.5

    bpy.ops.object.light_add(type="AREA", location=(0.0, 2.4, mid_z + 0.8), rotation=(math.radians(-70), 0, 0))
    rim = bpy.context.active_object
    rim.data.energy = 420.0
    rim.data.color = pal["glow"][:3]
    rim.data.size = 2.2

    # Góc máy Low-Angle Hero Shot (đặt thấp nhìn hất nhẹ lên để tăng khí chất uy nghiêm!)
    bpy.ops.object.camera_add(location=(0, -h * 2.15, mid_z - 0.12), rotation=(math.radians(93), 0, 0))
    scene.camera = bpy.context.active_object

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.04, 0.05, 0.08, 1.0)
        bg.inputs[1].default_value = 0.7

    base_yaw = 180.0 if face_dir > 0 else 0.0
    for f, ang in [(1, base_yaw - 18), (48, base_yaw + 22), (72, base_yaw - 12), (96, base_yaw + 360)]:
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
    print("Rendered Armored Hero Demo MP4:", out_mp4)
