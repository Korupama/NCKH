# Tài liệu tối ưu hóa và triển khai Slurm/Docker trên cụm NVIDIA H200

Tài liệu kỹ thuật mô tả các phương pháp tối ưu hiệu năng tính toán và quy trình triển khai container/Slurm cho pipeline đánh giá việt vị 9 stage trên phần cứng NVIDIA H200 (kiến trúc Hopper, 141 GB HBM3e VRAM).

## 1. Cơ sở tối ưu hóa (Bảo toàn độ chính xác số học)

Các kỹ thuật tối ưu hóa tập trung vào việc giảm thiểu overhead I/O, thời gian khởi tạo tiến trình và tận dụng phần cứng GPU mà không làm thay đổi kết quả số học của mô hình.

| Thành phần | Trước tối ưu | Sau tối ưu | Độ tương đương kết quả |
|---|---|---|---|
| Khởi tạo model | Mỗi frame khởi động tiến trình Python mới, nạp lại weights từ đĩa. | Giữ model thường trực trên VRAM/RAM (`Resident GPU Caching`). | Trọng số và kết quả forward giữ nguyên. |
| Stage 4 (SAM-3D Body) | Suy luận tuần tự từng bounding box (`--person-batch-size 1`). | Batching toàn bộ cầu thủ trên GPU (`--person-batch-size 16`). | Kết quả 3D keypoints độc lập từng người. |
| Stage 3 (RTMW ONNX) | Khởi tạo lại ONNX session mỗi frame. | Chuẩn hóa backend CUDA/CPU fallback và tái sử dụng session. | Tọa độ keypoints 2D giữ nguyên. |
| Stage 5, 7, 8, 9 | Gọi qua `subprocess.run` 5–6 lần/frame. | Gọi trực tiếp hàm trong cùng tiến trình (In-process execution). | Schema và dữ liệu JSON khớp 100%. |
| Quản lý bộ nhớ GPU | Gọi `torch.cuda.empty_cache()` sau mỗi stage. | Giữ allocator pool liên tục, tắt flush cache mặc định. | Không ảnh hưởng giá trị tensor. |
| Precision & cuDNN | Mặc định FP32 chuẩn. | Bật `torch.backends.cudnn.benchmark = True` và TensorFloat-32 (`high`). | Giữ nguyên dynamic range của FP32. |

---

## 2. Thông số phân vùng trên hệ thống HPC-UIT

Cụm máy chủ HPC-UIT cấu hình GPU H200 theo các MIG profile:

| Partition | MIG Profile | VRAM | CPU | RAM | Mục đích |
|---|---|---|---|---|---|
| `small` | `1g.18gb` | ~16 GiB | 12 | 40 GB | Thử nghiệm, debug frame đơn lẻ |
| `normal` | `2g.35gb` | ~32.5 GiB | 16 | 72 GB | Mặc định cho toàn bộ pipeline (~15 GB VRAM) |
| `large` | `3g.71gb` | ~70 GiB | 24 | 112 GB | Tác vụ yêu cầu VRAM lớn hơn 35 GB |

### Ràng buộc tài nguyên cụm

- Quota thư mục `/home/<username>`: 10 GB. Dữ liệu thí nghiệm, model checkpoints và cache Apptainer phải trỏ về `/datastore/<username>`.
- Giới hạn job: Tối đa 3 job chạy đồng thời trên mỗi tài khoản.
- Giãn cách submit: 1 giờ giữa các lệnh `sbatch`. Dùng Job Array để gửi 1 lệnh chia thành nhiều task chạy đồng thời.

---

## 3. Quy trình triển khai

### Bước 1: Build Docker image tại máy local

```bash
# Build Docker image chuẩn hóa
bash scripts/build_container.sh docker

# Xuất ra file tar
docker save nckh-pipeline:h200 -o nckh_h200.tar

# Chuyển lên datastore của server
scp nckh_h200.tar <username>@hpc.uit.edu.vn:/datastore/<username>/
```

### Bước 2: Chuẩn bị môi trường trên server

```bash
# Đăng nhập SSH
ssh <username>@hpc.uit.edu.vn

# Tạo thư mục làm việc trên datastore
mkdir -p /datastore/$USER
cd /datastore/$USER

# Clone source code
git clone https://github.com/Korupama/NCKH.git NCKH_SoICT
cd NCKH_SoICT
git checkout feature/h200-speedup-docker-slurm

# Cấu hình biến môi trường chuyển cache ra datastore
bash scripts/setup_uit_environment.sh
source ~/.bashrc

# Build file SIF
cd /datastore/$USER
apptainer build nckh_h200.sif docker-archive:///datastore/$USER/nckh_h200.tar
rm /datastore/$USER/nckh_h200.tar
```

### Bước 3: Gửi job Slurm

```bash
cd /datastore/$USER/NCKH_SoICT

# Chạy song song 3 worker qua Slurm Array
sbatch slurm/eval_uit_array.slurm

# Chạy cascaded benchmark song song 3 worker
EVAL_MODE=cascaded sbatch slurm/eval_uit_array.slurm

# Chạy 1 job đơn lẻ
sbatch slurm/eval_uit_normal.slurm
```

---

## 4. Danh mục file cấu hình và script

- `Dockerfile`: Image hoàn chỉnh dựa trên NVIDIA NGC PyTorch 24.04 (CUDA 12.4, cuDNN 9, SM 9.0a).
- `Dockerfile.light`: Bản image rút gọn dựa trên PyTorch Docker Hub (~4.5 GB).
- `.dockerignore`: Loại bỏ virtualenv, git objects, checkpoints khỏi build context.
- `scripts/build_container.sh`: Script tự động build Docker image và file Apptainer `.sif`.
- `scripts/setup_uit_environment.sh`: Cấu hình tự động đường dẫn cache và datastore trên cụm UIT.
- `slurm/eval_uit_array.slurm`: Cấu hình Slurm Job Array 3 worker song song (`--array=0-2%3`) cho partition `normal`.
- `slurm/eval_uit_normal.slurm`: Cấu hình Slurm cho 1 job đơn lẻ trên partition `normal`.
- `slurm/eval_uit_small.slurm`: Cấu hình Slurm chạy debug trên partition `small`.
- `video_bookmark/run_accelerated_pipeline.py`: Pipeline runner hỗ trợ nạp model thường trực, batching Stage 4 và chia chunk dữ liệu (`--num-chunks`, `--chunk-index`).
- `evaluate_cascaded_pipeline.py`: Đánh giá benchmark Stage 2 -> 5 -> 7 -> 8 với hỗ trợ chia chunk đánh giá phân tán.
