import gc
import os
import subprocess
import sys
import sysconfig
import traceback
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh
from PIL import Image

for _pkg in ("pytorch_lightning", "fast_simplification", "pybind11"):
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

import fast_simplification as _fs
_orig_simplify = _fs.simplify
def _safe_simplify(points, triangles, target_reduction=None, *args, **kwargs):
    n_faces = len(triangles)
    min_faces = 40000
    if n_faces <= min_faces:
        return points, triangles
    if target_reduction is not None:
        val = float(target_reduction)
        if val >= 1.0:
            # Caller passed target_count into target_reduction position
            target_count = max(min_faces, int(val))
            if n_faces <= target_count:
                return points, triangles
            val = 1.0 - (target_count / n_faces)
        elif val <= 0.0:
            return points, triangles
        else:
            max_red = max(0.0, 1.0 - (min_faces / n_faces))
            val = min(val, max_red)
            if val <= 0.01:
                return points, triangles
        return _orig_simplify(points, triangles, val, *args, **kwargs)
    return _orig_simplify(points, triangles, *args, **kwargs)
_fs.simplify = _safe_simplify


def ensure_mesh_inpaint_compiled():
    """Biên dịch mesh_inpaint_processor.cpp đúng phiên bản Python đang chạy (vd: Python 3.13 trên Colab)."""
    dr_dir = "/content/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    target_so = os.path.join(dr_dir, f"mesh_inpaint_processor{ext_suffix}")
    cpp_file = os.path.join(dr_dir, "mesh_inpaint_processor.cpp")
    if os.path.exists(target_so) or not os.path.exists(cpp_file):
        return

    inc_py = sysconfig.get_paths()["include"]
    if not os.path.exists(os.path.join(inc_py, "Python.h")):
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"Đang cài python{py_ver}-dev để biên dịch mesh_inpaint_processor...", flush=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", f"python{py_ver}-dev"], check=False)

    import pybind11
    inc_pb = pybind11.get_include()
    cmd = [
        "g++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
        f"-I{inc_py}", f"-I{inc_pb}",
        cpp_file, "-o", target_so,
    ]
    print("Đang biên dịch mesh_inpaint_processor cho Python", sys.version.split()[0], "...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(target_so):
        print("-> Biên dịch thành công:", target_so, flush=True)
    else:
        print("⚠️ Không biên dịch được C++ mesh_inpaint_processor, sẽ dùng SciPy fallback:", res.stderr[-300:], flush=True)


ensure_mesh_inpaint_compiled()

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
        scale = float(outscale) if outscale else 1.0
        if scale <= 1.0:
            return img, None
        if isinstance(img, np.ndarray):
            import cv2
            h, w = img.shape[:2]
            up = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LANCZOS4)
            return up, None
        if isinstance(img, Image.Image):
            w, h = img.size
            return img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS), None
        return img, None

    def to(self, *args, **kwargs):
        return self


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

# Nạp C++ mesh_inpaint_processor; nếu lỗi mới chèn fallback pure-SciPy vào sys.modules
try:
    import DifferentiableRenderer.mesh_inpaint_processor as _mip  # type: ignore
    print("-> Đã nạp C++ mesh_inpaint_processor thành công!", flush=True)
except Exception:
    from scipy.ndimage import distance_transform_edt

    def _scipy_mesh_vertice_inpaint(texture, mask, *args, **kwargs):
        tex = np.asarray(texture).copy()
        msk = np.asarray(mask)
        if msk.ndim == 3:
            msk = msk[..., 0]
        invalid = (msk == 0)
        if invalid.any() and (~invalid).any():
            indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
            tex[invalid] = tex[tuple(indices[:, invalid])]
        return tex

    _fake_mip = types.ModuleType("DifferentiableRenderer.mesh_inpaint_processor")
    _fake_mip.meshVerticeInpaint = _scipy_mesh_vertice_inpaint
    sys.modules["DifferentiableRenderer.mesh_inpaint_processor"] = _fake_mip
    sys.modules["mesh_inpaint_processor"] = _fake_mip
    print("-> Đã kích hoạt SciPy fallback cho mesh_inpaint_processor", flush=True)

try:
    import DifferentiableRenderer.mesh_utils as _mu
    if not hasattr(sys.modules.get("bpy"), "ops"):
        def _trimesh_obj_to_glb(obj_path, glb_path, **kwargs):
            trimesh.load(obj_path, process=False).export(glb_path)
        _mu.convert_obj_to_glb = _trimesh_obj_to_glb
except Exception:
    pass

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

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    num_views = 4 if vram_gb < 18 else 6
    print(f"3/3 Đang tô màu PBR 3D (Hunyuan3D Paint | VRAM={vram_gb:.1f}GB | views={num_views})...", flush=True)
    try:
        import inspect
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        conf = Hunyuan3DPaintConfig(max_num_view=num_views, resolution=512)
        paint_pipe = Hunyuan3DPaintPipeline(conf)
        sig = inspect.signature(paint_pipe.__call__).parameters
        painted_obj = "/content/Test/assets/models_3d/hoa_do_painted.obj"
        if "mesh_path" in sig:
            out_res = paint_pipe(mesh_path=shape_obj, image_path=clean_png, output_mesh_path=painted_obj)
        else:
            out_res = paint_pipe(shape_obj, clean_png)

        out_path = out_res if isinstance(out_res, str) and os.path.exists(out_res) else painted_obj
        painted_glb = os.path.splitext(out_path)[0] + ".glb"
        if os.path.exists(painted_glb):
            import shutil
            shutil.copy2(painted_glb, raw_glb)
            print("-> Đã sao chép GLB có màu PBR:", raw_glb, flush=True)
        elif os.path.exists(out_path):
            trimesh.load(out_path, process=False).export(raw_glb)
            print("-> Đã xuất GLB từ OBJ có màu PBR (giữ nguyên UV):", raw_glb, flush=True)
        else:
            mesh.export(raw_glb)
        del paint_pipe
    except Exception as exc:
        print(f"⚠️ Paint PBR lỗi ({exc}):", flush=True)
        traceback.print_exc()
        mesh.export(raw_glb)

    gc.collect()
    torch.cuda.empty_cache()
    print("Raw 3D GLB:", raw_glb, f"({os.path.getsize(raw_glb) / 1024 / 1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
