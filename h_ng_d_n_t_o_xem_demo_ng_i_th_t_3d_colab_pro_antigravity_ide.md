# Quy Trình Tạo Mô Hình 3D Người Thật Liên Tục (`.GLB` / `.FBX`) Với Google Colab Pro & Antigravity IDE

Tài liệu này hướng dẫn quy trình khép kín, không bị giới hạn lượt miễn phí hay lỗi timeout ZeroGPU: **Ảnh A-Pose → Hunyuan3D-2.1 (Colab Pro) → Gắn xương (AccuRIG) → Tối ưu (Blender) → Xem demo trực tiếp trên Antigravity IDE**.

---

## 1. Bước 1: Chuẩn bị ảnh mẫu người thật dáng A-Pose

Ảnh đầu vào quyết định phần lớn độ nét khuôn mặt và tỷ lệ khớp khi gắn xương.

### Tiêu chí bắt buộc:
* **Toàn thân từ đỉnh đầu đến gót giày**, đứng thẳng góc chính diện.
* **Dáng A-Pose đối xứng:** Hai tay dang góc ~45° tách hẳn khỏi hông, ngón tay hơi mở, hai chân dang rộng bằng vai.
* **Nền trơn sáng, ánh sáng phẳng (studio lighting)**, không bóng đổ gắt, không cầm vật thể trên tay.

### Mẫu Prompt chuẩn (thay nội dung trong `[...]`):
```text
Full body photorealistic studio portrait of a [25-year-old Vietnamese man/woman, short black hair, wearing a fitted navy jacket, dark jeans, and sneakers], standing straight facing directly forward in a symmetrical A-pose, arms relaxed at 45 degrees away from body, fingers slightly separated, feet shoulder-width apart. Flat even studio lighting, no harsh shadows, plain solid light gray background, sharp focus, 8k, realistic skin texture, character design reference sheet.
```

---

## 2. Bước 2: Chạy Hunyuan3D-2.1 trên Google Colab Pro

### Chọn cấu hình GPU (`Runtime → Change runtime type`):

| Loại GPU | VRAM | Khuyến nghị sử dụng | Cờ tối ưu |
| :--- | :--- | :--- | :--- |
| **L4** | 24 GB | Tiết kiệm Compute Units nhất, chạy ổn định cho vài nhân vật/lần | Giữ `--low_vram_mode` |
| **A100** | 40 GB | Nhanh nhất, phù hợp khi gom sẵn 10–30 ảnh chạy dồn 1 lượt | Có thể bỏ `--low_vram_mode` |

### Tạo Notebook mới và chạy lần lượt 3 Cell sau:

#### Cell 1 — Kết nối Google Drive và tải mã nguồn
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
%cd /content/Hunyuan3D-2.1
```

#### Cell 2 — Cài thư viện và biên dịch bộ lên màu PBR (mất ~3–5 phút)
```bash
# Cài các thư viện phụ thuộc
!pip install -q -r requirements.txt

# Biên dịch bộ render texture cho bước tô màu
%cd /content/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
!pip install -q -e .

%cd /content/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
!bash compile_mesh_painter.sh

# Cài cloudflared để tạo link giao diện web riêng
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

%cd /content/Hunyuan3D-2.1
```
> **Xử lý lỗi xung đột (nếu có):** Nếu `pip install -r requirements.txt` báo lỗi xung đột phiên bản `torch` hoặc `bpy`, hãy mở file `requirements.txt`, thêm dấu `#` trước các dòng `torch`, `torchvision`, `torchaudio` để giữ nguyên bộ PyTorch/CUDA có sẵn của Colab rồi chạy lại Cell 2.

#### Cell 3 — Khởi chạy giao diện web riêng không giới hạn lượt
```python
import subprocess, time, re

log_file = open("tunnel.log", "w")
subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:7860"],
    stdout=log_file,
    stderr=log_file
)
time.sleep(4)

with open("tunnel.log") as f:
    urls = re.findall(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", f.read())
print(">>> Link giao diện 3D của bạn:", urls[0] if urls else "Đang khởi tạo, kiểm tra file tunnel.log")

# Lần đầu chạy sẽ tự tải trọng số model (~2-3 phút)
# Nếu dùng A100 40GB, có thể xóa cờ --low_vram_mode để chạy nhanh hơn
!python gradio_app.py --host 127.0.0.1 --port 7860 --low_vram_mode
```

### Thao tác xuất `.glb` trên giao diện web:
1. Bấm vào link `https://...trycloudflare.com` vừa in ra ở Cell 3.
2. Kéo ảnh A-Pose vào ô **Image Prompt** → bấm **`Gen Textured Shape`** (tạo cả lưới lẫn màu PBR).
3. Ở tab **Export**: giữ `File Type` là `glb`. Nếu muốn giữ mặt nét nhất thì **bỏ tích** `Simplify Mesh` (hoặc tích và đặt `Target Face Number` khoảng `50000`–`80000`).
4. Bấm **`Transform`** → bấm **`Download`** để tải file `.glb`.

> **Mẹo tiết kiệm Compute Units:** Chuẩn bị sẵn toàn bộ ảnh trước khi mở GPU. Tải hết file `.glb` xong phải vào **`Runtime → Disconnect and delete runtime`** ngay để tránh bị trừ giờ chạy ngầm.

---

## 3. Bước 3: Gắn khung xương (AccuRIG) & Tối ưu file (Blender)

Toàn bộ bước này chạy offline trên máy tính, miễn phí vĩnh viễn và không giới hạn số lượng.

### A. Gắn xương tự động bằng ActorCore AccuRIG
1. Kéo thả file `.glb` vào cửa sổ **AccuRIG** → bấm **`Rig Body`**.
2. Căn nhẹ các điểm khớp (cổ, vai, khuỷu tay, cổ tay, háng, đầu gối, mắt cá) vào đúng trục thân người (bật **Symmetry** để tự đối xứng 2 bên).
3. Bấm **`Rig Right Hand`** → chọn 5 ngón → căn nhanh khớp tay → copy sang **`Left Hand`** → bấm **`Finalize Character`**.
4. Xem chuyển động test thử → bấm **`Export FBX`** (chọn Target là *Blender*, *Unity*, hoặc *Unreal*).
5. *(Tùy chọn)* Tải file `.fbx` vừa xuất lên **Adobe Mixamo** (`mixamo.com`) để gắn thêm các động tác (đi bộ, chạy, nói chuyện...) rồi tải về ở chế độ **With Skin**.

### B. Tối ưu độ bóng da và xuất bộ đôi `.GLB` + `.FBX` trong Blender
1. Mở **Blender**, xóa khối hộp mặc định (`A` → `X` → `Delete`) → `File → Import → FBX (.fbx)` (hoặc `.glb` nếu không cần xương).
2. **Sửa lỗi bề mặt bóng nhựa:** Vào tab **Shading**, chọn nhân vật → tìm node **Principled BSDF** → chỉnh `Metallic = 0` và `Roughness = 0.55 – 0.75`.
3. **Giảm lưới nếu file quá nặng (> 15 MB):** Chọn Mesh → mở bảng **Modifiers** (hình cờ lê) → `Generate → Decimate` → đặt `Ratio = 0.35 – 0.5` → bấm **Apply**.
4. **Xuất file chuẩn:**
   * **Xuất `.glb`:** `File → Export → glTF 2.0 (.glb/.gltf)` → Format `glTF Binary (.glb)` → tích **`Compression`** (nén Draco) và giữ tích **Animation** nếu có xương.
   * **Xuất `.fbx`:** `File → Export → FBX (.fbx)` → đổi **Path Mode** sang **`Copy`** và **bấm bật nút hình chiếc hộp (Embed Textures)** bên cạnh để không bị mất màu.

### C. Script xử lý hàng loạt không cần mở giao diện Blender (`batch_export.py`)
Lưu đoạn mã sau thành file `batch_export.py`:

```python
import bpy, sys, os

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
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs['Metallic'].default_value = 0.0
                if node.inputs['Roughness'].default_value < 0.5:
                    node.inputs['Roughness'].default_value = 0.65

bpy.ops.export_scene.gltf(
    filepath=os.path.join(out_dir, f"{base_name}_ready.glb"),
    export_format='GLB',
    export_draco_mesh_compression_enable=True
)

bpy.ops.export_scene.fbx(
    filepath=os.path.join(out_dir, f"{base_name}_ready.fbx"),
    path_mode='COPY',
    embed_textures=True
)
```

Chạy trực tiếp qua terminal:
```bash
blender --background --python batch_export.py -- "duong_dan/nhan_vat.fbx" "thu_muc_xuat"
```

---

## 4. Bước 4: Xem demo file `.GLB` / `.FBX` trực tiếp trên Google Antigravity IDE

**Có thể xem demo trực tiếp ngay trong Antigravity IDE** mà không cần thoát ra phần mềm khác. Tùy mục đích kiểm tra nhanh hay test tích hợp vào dự án, bạn chọn 1 trong 3 cách sau:

### Cách 1: Cài Extension xem trực tiếp ngay trong tab Editor (Nhanh nhất)
Mở mục **Extensions** (`Ctrl + Shift + X`) trong Antigravity IDE và cài 1 trong 3 extension sau (đều có sẵn trên Open VSX):

| Tên Extension | Định dạng hỗ trợ | Điểm mạnh nổi bật |
| :--- | :--- | :--- |
| **Nightmesh — 3D Model Viewer** | `.glb`, `.gltf` | Hỗ trợ file đã **nén Draco/Meshopt/KTX2**, chạy thử animation, xem cây khung xương/chất liệu, có nút **Copy Reference** trỏ thẳng bộ phận 3D vào khung chat AI Agent. |
| **Super GLB Viewer** | `.glb` | Đổi qua lại 4 engine hiển thị (**Three.js, Babylon.js, Unity, PlayCanvas**), test animation & blendshapes khuôn mặt, xem thông số tam giác và VRAM. |
| **3D Model Viewer (Online3DViewer)** | `.glb`, `.fbx`, `.obj`, `.stl` | Mở được cả **`.fbx` lẫn `.glb`**, tự động reload khi bạn xuất đè file mới *(lưu ý: chưa mở được `.glb` đã nén Draco)*. |

*Sau khi cài xong:* Bấm trực tiếp vào file `.glb` hoặc `.fbx` trong cây thư mục bên trái để xoay 360°, zoom, kiểm tra ánh sáng và bấm Play chạy thử animation ngay trong IDE. (Nếu file mở ra dạng chữ lỗi mã hóa, chuột phải vào file → **Open With…** → chọn extension vừa cài).

### Cách 2: Nhờ Agent trong Antigravity IDE dựng trang Preview Web (`<model-viewer>` / Three.js)
Phù hợp khi bạn muốn xem thử mô hình chạy thực tế trên trình duyệt, chỉnh ánh sáng HDRI, test đổi nhiều animation hoặc kiểm tra FPS:
1. Chép file `.glb` vào thư mục dự án (ví dụ: `public/models/character.glb`).
2. Nhắn trực tiếp cho Agent trong Antigravity IDE:
   > *"Tạo giúp mình file `preview.html` dùng Google `<model-viewer>` (hoặc Three.js) để load và xem thử file `models/character.glb`, có nút chuyển đổi các animation clip có sẵn trong file và mở local server để preview."*
3. Mở trang preview ngay bằng trình duyệt tích hợp của IDE để kiểm tra thực tế.

### Cách 3: Kết nối Blender trực tiếp qua MCP (Nâng cao)
Nếu bạn thường xuyên cần sửa lưới, đổi vật liệu hoặc render ảnh/video preview từ file `.glb` / `.fbx` ngay trong luồng làm việc với AI, có thể cài **Blender MCP Server** vào file cấu hình MCP (`mcp_config.json`) của Antigravity IDE để ra lệnh trực tiếp cho Blender xử lý mô hình mà không phải chuyển cửa sổ thủ công.