import gc
import os
import subprocess
import sys
import types
import torch
import trimesh
from PIL import Image

for _pkg in ("pytorch_lightning", "fast_simplification"):
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

import fast_simplification as _fs
_orig_simplify = _fs.simplify
def _safe_simplify(points, triangles, target_reduction=0.5, *args, **kwargs):
    if target_reduction <= 0:
        return points, triangles
    target_reduction = min(0.95, float(target_reduction))
    return _orig_simplify(points, triangles, target_reduction, *args, **kwargs)
_fs.simplify = _safe_simplify

sys.path.insert(0, "/content/Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, "/content/Hunyuan3D-2.1/hy3dpaint")
os.chdir("/content/Hunyuan3D-2.1")

try:
    import bpy  # type: ignore
except ImportError:
    sys.modules["bpy"] = types.ModuleType("bpy")

class _DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def enhance(self, img, outscale=4, **kwargs):
        return img, None
    def to(self, *args, **kwargs):
        return self

# Nếu thiếu basicsr/realesrgan trên Python 3.13, tạo stub an toàn để không chặn import
for _mod in ("basicsr", "basicsr.archs", "basicsr.archs.rrdbnet_arch", "realesrgan"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            m = types.ModuleType(_mod)
            if _mod.endswith("rrdbnet_arch"):
                m.RRDBNet = _DummyModel
            if _mod == "realesrgan":
                m.RealESRGANer = _DummyModel
            sys.modules[_mod] = m

import DifferentiableRenderer.mesh_utils as _mu
if not hasattr(sys.modules.get("bpy"), "ops"):
    def _trimesh_obj_to_glb(obj_path, glb_path, **kwargs):
        trimesh.load(obj_path).export(glb_path)
    _mu.convert_obj_to_glb = _trimesh_obj_to_glb

# Tự động ánh xạ key Dinov2Attention trên mọi phiên bản transformers
_orig_module_load_sd = torch.nn.Module.load_state_dict

def _compat_load_sd(self, state_dict, strict=True, assign=False):
    if self.__class__.__name__ == "SingleImageEncoder":
        expected = set(self.state_dict().keys())
        if any(".attention.q_proj." in k for k in expected) and any(".attention.attention.query." in k for k in state_dict):
            state_dict = {
                k.replace(".attention.attention.query.", ".attention.q_proj.")
                .replace(".attention.attention.key.", ".attention.k_proj.")
                .replace(".attention.attention.value.", ".attention.v_proj.")
                .replace(".attention.output.dense.", ".attention.o_proj."): v
                for k, v in state_dict.items()
            }
    return _orig_module_load_sd(self, state_dict, strict=strict, assign=assign)

torch.nn.Module.load_state_dict = _compat_load_sd

# Tự động bật trust_remote_code=True cho custom pipeline hunyuanpaintpbr trên diffusers >= 0.35
import diffusers
_orig_dp_from_pretrained = diffusers.DiffusionPipeline.from_pretrained.__func__

@classmethod
def _compat_dp_from_pretrained(cls, *args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    return _orig_dp_from_pretrained(cls, *args, **kwargs)

diffusers.DiffusionPipeline.from_pretrained = _compat_dp_from_pretrained

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def main():
    raw_glb = "/content/Test/assets/models_3d/hoa_do_raw.glb"
    shape_obj = "/content/Test/assets/models_3d/hoa_do_shape.obj"
    apose_png = "/content/Test/assets/apose_inputs/hoa_do_apose.png"
    os.makedirs("/content/Test/assets/models_3d", exist_ok=True)
    if os.path.exists(raw_glb) and os.path.getmtime(raw_glb) >= os.path.getmtime(apose_png):
        print("Skip (đã có):", raw_glb, flush=True)
        return

    print("1/3 Đang tách nền ảnh A-Pose...", flush=True)
    img = Image.open(apose_png).convert("RGBA")
    img = BackgroundRemover()(img)
    clean_png = "/content/Test/assets/models_3d/hoa_do_clean.png"
    img.save(clean_png)

    if not os.path.exists(shape_obj):
        print("2/3 Đang tạo hình khối 3D (Hunyuan3D Shape)...", flush=True)
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
        mesh = shape_pipe(image=img)[0]
        mesh.export(shape_obj)
        del shape_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("2/3 Dùng lại shape 3D đã tạo:", shape_obj, flush=True)
        mesh = trimesh.load(shape_obj)

    print("3/3 Đang tô màu PBR 3D (Hunyuan3D Paint)...", flush=True)
    try:
        import inspect
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        paint_pipe = Hunyuan3DPaintPipeline(conf)
        sig = inspect.signature(paint_pipe.__call__).parameters
        painted_obj = "/content/Test/assets/models_3d/hoa_do_painted.obj"
        if "mesh_path" in sig:
            out_res = paint_pipe(mesh_path=shape_obj, image_path=clean_png, output_mesh_path=painted_obj)
        else:
            out_res = paint_pipe(shape_obj, clean_png)
        out_path = out_res if isinstance(out_res, str) and os.path.exists(out_res) else painted_obj
        if os.path.exists(out_path):
            trimesh.load(out_path).export(raw_glb)
        else:
            mesh.export(raw_glb)
        del paint_pipe
    except Exception as exc:
        print(f"⚠️ Paint PBR lỗi ({exc}), xuất mesh 3D gốc để Blender gắn xương...", flush=True)
        mesh.export(raw_glb)

    gc.collect()
    torch.cuda.empty_cache()
    print("Raw 3D GLB:", raw_glb, flush=True)


if __name__ == "__main__":
    main()
