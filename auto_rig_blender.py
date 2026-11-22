"""
Headless Blender script:
1. Imports a raw A-Pose .glb (e.g. from Hunyuan3D-2.1)
2. Normalizes height to 1.75m standing on Z=0
3. Builds a humanoid Armature and binds with Automatic Weights
4. Fixes PBR materials (Metallic=0, Roughness=0.65)
5. Adds a simple heroic fighting-idle + punch animation
6. Exports <name>_rigged.glb and renders a 4-second 3D turntable/action preview .mp4
"""
import math
import os
import sys
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
input_glb = os.path.abspath(argv[0])
out_glb = os.path.abspath(argv[1])
out_mp4 = os.path.abspath(argv[2]) if len(argv) > 2 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_glb)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError(f"No mesh found in {input_glb}")

# Join all meshes into one character mesh
bpy.ops.object.select_all(action="DESELECT")
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
char_obj = bpy.context.active_object
char_obj.name = "HeroMesh"

# Apply transforms and normalize height to 1.75m standing on Z=0
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bbox = [char_obj.matrix_world @ Vector(c) for c in char_obj.bound_box]
min_z = min(v.z for v in bbox)
max_z = max(v.z for v in bbox)
height = max(0.01, max_z - min_z)
scale_f = 1.75 / height
char_obj.scale = (scale_f, scale_f, scale_f)
bpy.ops.object.transform_apply(scale=True)

bbox = [char_obj.matrix_world @ Vector(c) for c in char_obj.bound_box]
min_z = min(v.z for v in bbox)
cx = 0.5 * (min(v.x for v in bbox) + max(v.x for v in bbox))
cy = 0.5 * (min(v.y for v in bbox) + max(v.y for v in bbox))
char_obj.location = (-cx, -cy, -min_z)
bpy.ops.object.transform_apply(location=True)

# Làm sạch đỉnh trùng & pháp tuyến + giảm lưới (<45k mặt) để ARMATURE_AUTO bám xương chính xác
bpy.context.view_layer.objects.active = char_obj
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.002)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")

if len(char_obj.data.polygons) > 45000:
    dec = char_obj.modifiers.new(name="Decimate", type="DECIMATE")
    dec.ratio = 45000.0 / len(char_obj.data.polygons)
    bpy.context.view_layer.objects.active = char_obj
    bpy.ops.object.modifier_apply(modifier=dec.name)

# Fix PBR plastic shine
for mat in bpy.data.materials:
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Metallic"].default_value = 0.0
                if node.inputs["Roughness"].default_value < 0.5:
                    node.inputs["Roughness"].default_value = 0.65

# Build Humanoid Armature scaled to 1.75m A-Pose
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm_obj = bpy.context.active_object
arm_obj.name = "HeroRig"
arm = arm_obj.data
arm.edit_bones.remove(arm.edit_bones[0])

def add_bone(name, head, tail, parent=None):
    b = arm.edit_bones.new(name)
    b.head = Vector(head)
    b.tail = Vector(tail)
    if parent:
        b.parent = parent
    return b

hips = add_bone("Hips", (0, 0, 0.90), (0, 0, 1.05))
spine = add_bone("Spine", (0, 0, 1.05), (0, 0, 1.25), hips)
chest = add_bone("Chest", (0, 0, 1.25), (0, 0, 1.45), spine)
neck = add_bone("Neck", (0, 0, 1.45), (0, 0, 1.54), chest)
head = add_bone("Head", (0, 0, 1.54), (0, 0, 1.75), neck)

for side, s in [("L", 1), ("R", -1)]:
    sh = add_bone(f"Shoulder_{side}", (0.05 * s, 0, 1.40), (0.16 * s, 0, 1.38), chest)
    ua = add_bone(f"UpperArm_{side}", (0.16 * s, 0, 1.38), (0.34 * s, 0, 1.18), sh)
    fa = add_bone(f"ForeArm_{side}", (0.34 * s, 0, 1.18), (0.50 * s, 0, 0.98), ua)
    hd = add_bone(f"Hand_{side}", (0.50 * s, 0, 0.98), (0.58 * s, 0, 0.90), fa)

    th = add_bone(f"Thigh_{side}", (0.10 * s, 0, 0.90), (0.12 * s, 0, 0.50), hips)
    kn = add_bone(f"Shin_{side}", (0.12 * s, 0, 0.50), (0.12 * s, 0, 0.08), th)
    ft = add_bone(f"Foot_{side}", (0.12 * s, 0, 0.08), (0.12 * s, -0.14, 0.02), kn)

bpy.ops.object.mode_set(mode="OBJECT")

# Parent mesh with automatic weights + nearest-bone fallback if heat weighting fails
bpy.ops.object.select_all(action="DESELECT")
char_obj.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# Kiểm tra xem ARMATURE_AUTO có gán được trọng số không; nếu rỗng thì gán theo xương gần nhất
has_weights = False
for vg in char_obj.vertex_groups:
    for v in char_obj.data.vertices[:min(200, len(char_obj.data.vertices))]:
        try:
            if vg.weight(v.index) > 0.01:
                has_weights = True
                break
        except RuntimeError:
            pass
    if has_weights:
        break

if not has_weights:
    print("⚠️ ARMATURE_AUTO không bám được weight, chuyển sang gán weight theo xương gần nhất...")
    from mathutils.geometry import intersect_point_line
    bone_segs = [(b.name, b.head_local.copy(), b.tail_local.copy()) for b in arm_obj.data.bones]
    for bname, _, _ in bone_segs:
        if bname not in char_obj.vertex_groups:
            char_obj.vertex_groups.new(name=bname)
    for v in char_obj.data.vertices:
        p = v.co
        best_name, best_d = "Hips", 1e9
        for bname, h_pt, t_pt in bone_segs:
            closest, fac = intersect_point_line(p, h_pt, t_pt)
            if fac < 0.0:
                closest = h_pt
            elif fac > 1.0:
                closest = t_pt
            d = (p - closest).length
            if d < best_d:
                best_d, best_name = d, bname
        char_obj.vertex_groups[best_name].add([v.index], 1.0, "REPLACE")

# Làm mượt trọng số xương để không bị khoét thân hoặc kéo tia nhọn ở khớp
bpy.context.view_layer.objects.active = char_obj
bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=0.5, repeat=4)
bpy.ops.object.mode_set(mode="OBJECT")

# Keyframe a 96-frame (4s @ 24fps) heroic idle + outward-clear punch animation
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 96
scene.render.fps = 24

bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="POSE")
pb = arm_obj.pose.bones
for pb_item in pb:
    pb_item.rotation_mode = "XYZ"

def key_pose(frame, poses):
    scene.frame_set(frame)
    for bname, rot_deg in poses.items():
        if bname in pb:
            pb[bname].rotation_euler = tuple(math.radians(a) for a in rot_deg)
            pb[bname].keyframe_insert(data_path="rotation_euler", frame=frame)

key_pose(1, {
    "Chest": (0, 0, 0), "Head": (0, 0, 0),
    "UpperArm_R": (0, 0, 0), "ForeArm_R": (0, 0, 0),
    "UpperArm_L": (0, 0, 0), "ForeArm_L": (0, 0, 0),
})
# Tư thế thủ chiến đấu: 2 tay mở nhẹ ra ngoài và hướng về trước, không chạm thân
key_pose(24, {
    "Chest": (3, 0, -5), "Head": (-3, 0, 5),
    "UpperArm_R": (-20, 0, 15), "ForeArm_R": (-32, 0, 0),
    "UpperArm_L": (-18, 0, -15), "ForeArm_L": (-28, 0, 0),
})
# Tung cú đấm thẳng hướng về trước (-Y) & hơi chếch ra ngoài
key_pose(48, {
    "Chest": (4, 0, 8), "Head": (-2, 0, -6),
    "UpperArm_R": (-48, 0, 18), "ForeArm_R": (-8, 0, 0),
    "UpperArm_L": (-15, 0, -20), "ForeArm_L": (-35, 0, 0),
})
key_pose(72, {
    "Chest": (3, 0, -5), "Head": (-3, 0, 5),
    "UpperArm_R": (-20, 0, 15), "ForeArm_R": (-32, 0, 0),
    "UpperArm_L": (-18, 0, -15), "ForeArm_L": (-28, 0, 0),
})
key_pose(96, {
    "Chest": (0, 0, 0), "Head": (0, 0, 0),
    "UpperArm_R": (0, 0, 0), "ForeArm_R": (0, 0, 0),
    "UpperArm_L": (0, 0, 0), "ForeArm_L": (0, 0, 0),
})

bpy.ops.object.mode_set(mode="OBJECT")

os.makedirs(os.path.dirname(out_glb), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    export_animations=True,
)
print("Exported rigged GLB:", out_glb)

if out_mp4:
    # Add lighting + orbiting camera for demo render
    bpy.ops.object.light_add(type="SUN", location=(3, -4, 5))
    bpy.context.active_object.data.energy = 3.5
    bpy.ops.object.light_add(type="AREA", location=(-3, -2, 3))
    bpy.context.active_object.data.energy = 120.0

    bpy.ops.object.camera_add(location=(0, -4.0, 0.92), rotation=(math.radians(88), 0, 0))
    cam = bpy.context.active_object
    scene.camera = cam
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.12, 0.14, 0.18, 1.0)
        bg.inputs[1].default_value = 1.2

    # Rotate hero slowly for 360-ish preview while punching
    for f, ang in [(1, -20), (48, 25), (96, 360)]:
        arm_obj.rotation_euler = (0, 0, math.radians(ang))
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=f)

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
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

    scene.render.resolution_x = 800
    scene.render.resolution_y = 450
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = out_mp4
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    bpy.ops.render.render(animation=True)
    print("Rendered 3D demo MP4:", out_mp4)
