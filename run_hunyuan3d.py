import gc
import os
import sys
import types
import torch
import trimesh
from PIL import Image

sys.path.insert(0, "/content/Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, "/content/Hunyuan3D-2.1/hy3dpaint")
os.chdir("/content/Hunyuan3D-2.1")

try:
    import bpy  # type: ignore
except ImportError:
    sys.modules["bpy"] = types.ModuleType("bpy")

import DifferentiableRenderer.mesh_utils as _mu
if not hasattr(sys.modules.get("bpy"), "ops"):
    def _trimesh_obj_to_glb(obj_path, glb_path, **kwargs):
        trimesh.load(obj_path).export(glb_path)
    _mu.convert_obj_to_glb = _trimesh_obj_to_glb

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.models.conditioners import SingleImageEncoder
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


# Tự động ánh xạ key Dinov2Attention khi chạy trên transformers >= 4.50
_orig_load_sd = SingleImageEncoder.load_state_dict

def _compat_load_sd(self, state_dict, strict=True, assign=False):
    expected = set(self.state_dict().keys())
    if any(".attention.q_proj." in k for k in expected) and any(".attention.attention.query." in k for k in state_dict):
        remapped = {}
        for k, v in state_dict.items():
            nk = (
                k.replace(".attention.attention.query.", ".attention.q_proj.")
                .replace(".attention.attention.key.", ".attention.k_proj.")
                .replace(".attention.attention.value.", ".attention.v_proj.")
                .replace(".attention.output.dense.", ".attention.o_proj.")
            )
            remapped[nk] = v
        state_dict = remapped
    return _orig_load_sd(self, state_dict, strict=strict, assign=assign)

SingleImageEncoder.load_state_dict = _compat_load_sd


def main():
    raw_glb = "/content/Test/assets/models_3d/hoa_do_raw.glb"
    os.makedirs("/content/Test/assets/models_3d", exist_ok=True)
    if os.path.exists(raw_glb):
        print("Skip (đã có):", raw_glb, flush=True)
        return

    print("1/3 Đang tách nền ảnh A-Pose...", flush=True)
    img = Image.open("/content/Test/assets/apose_inputs/hoa_do_apose.png").convert("RGBA")
    img = BackgroundRemover()(img)

    print("2/3 Đang tạo hình khối 3D (Hunyuan3D Shape)...", flush=True)
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    mesh = shape_pipe(image=img)[0]
    del shape_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("3/3 Đang tô màu PBR 3D (Hunyuan3D Paint)...", flush=True)
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_pipe = Hunyuan3DPaintPipeline(conf)
    mesh = paint_pipe(mesh, image=img)
    mesh.export(raw_glb)
    del paint_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Raw 3D GLB:", raw_glb, flush=True)


if __name__ == "__main__":
    main()
