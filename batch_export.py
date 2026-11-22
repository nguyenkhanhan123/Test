import bpy
import sys
import os

argv = sys.argv[sys.argv.index("--") + 1:]
input_path, out_dir = argv[0], argv[1]
base_name = os.path.splitext(os.path.basename(input_path))[0]

bpy.ops.wm.read_factory_settings(use_empty=True)

if input_path.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=input_path)
else:
    bpy.ops.import_scene.gltf(filepath=input_path)

for mat in bpy.data.materials:
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Metallic"].default_value = 0.0
                if node.inputs["Roughness"].default_value < 0.5:
                    node.inputs["Roughness"].default_value = 0.65

os.makedirs(out_dir, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=os.path.join(out_dir, f"{base_name}_ready.glb"),
    export_format="GLB",
    export_draco_mesh_compression_enable=True,
)

bpy.ops.export_scene.fbx(
    filepath=os.path.join(out_dir, f"{base_name}_ready.fbx"),
    path_mode="COPY",
    embed_textures=True,
)
