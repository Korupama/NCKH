# Hướng dẫn chạy pipeline trên hệ thống HPC-UIT (NVIDIA H200)

Tài liệu hướng dẫn vận hành pipeline 9-stage trên hệ thống tính toán hiệu năng cao của Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (`hpc.uit.edu.vn`).

## 1. Thông số cụm và tài nguyên

Hệ thống quản lý tài nguyên bằng Slurm Workload Manager. Node tính toán sử dụng GPU NVIDIA H200 phân chia theo kiến trúc Multi-Instance GPU (MIG).

| Partition | MIG Profile | VRAM | CPU | RAM | Thời gian tối đa | Sử dụng |
|---|---|---|---|---|---|---|
| `small` | `1g.18gb` | ~16 GiB | 12 | 40 GB | 72:00:00 | Debug, chạy thử đơn lẻ |
| `normal` | `2g.35gb` | ~32.5 GiB | 16 | 72 GB | 72:00:00 | Mặc định cho toàn bộ pipeline |
| `large` | `3g.71gb` | ~70 GiB | 24 | 112 GB | 72:00:00 | Chạy tải lớn |

### Quy định của hệ thống

1. Thư mục `/home/<username>` có quota giới hạn 10 GB. Quá giới hạn này hệ thống sẽ khóa quyền submit job. Tất cả code, dữ liệu, weights, logs và Apptainer cache phải đặt tại `/datastore/<username>`.
2. Giới hạn tối đa 3 job chạy đồng thời trên mỗi tài khoản.
3. Giãn cách thời gian giữa các lần gọi lệnh submit `sbatch` là 1 giờ. Khi cần chạy song song nhiều worker, sử dụng Slurm Job Array (`--array=0-2%3`) để gửi 1 lệnh duy nhất.

---

## 2. Chuẩn bị container

Thực hiện trên máy cá nhân có Docker Engine:

```bash
cd NCKH_SoICT

# Build Docker image (chọn 1 trong 2 cấu hình)
bash scripts/build_container.sh docker   # Bản NGC PyTorch 24.04 (mặc định)
# bash scripts/build_container.sh light  # Bản gọn nhẹ 4.5 GB

# Xuất image ra file tar
docker save nckh-pipeline:h200 -o nckh_h200.tar

# Chuyển file tar vào /datastore trên server trường
scp nckh_h200.tar <username>@hpc.uit.edu.vn:/datastore/<username>/
```

---

## 3. Thiết lập trên server HPC-UIT

Đăng nhập qua SSH (bật VPN UIT nếu ở ngoài mạng trường):

```bash
ssh <username>@hpc.uit.edu.vn
```

Thiết lập thư mục làm việc và chuyển hướng cache:

```bash
# Khởi tạo workspace trên datastore
mkdir -p /datastore/$USER
cd /datastore/$USER

# Clone source code
git clone https://github.com/Korupama/NCKH.git NCKH_SoICT
cd NCKH_SoICT
git checkout feature/h200-speedup-docker-slurm

# Cấu hình biến môi trường chuyển cache ra datastore
bash scripts/setup_uit_environment.sh
source ~/.bashrc
```

Đóng gói file container `.sif`:

```bash
cd /datastore/$USER
apptainer build nckh_h200.sif docker-archive:///datastore/$USER/nckh_h200.tar
rm /datastore/$USER/nckh_h200.tar
```

---

## 4. Gửi job tính toán

Di chuyển vào thư mục dự án:

```bash
cd /datastore/$USER/NCKH_SoICT
```

### Chạy song song 3 worker (Slurm Job Array)

Tận dụng giới hạn 3 job song song, chia đều dữ liệu cho 3 worker:

```bash
sbatch slurm/eval_uit_array.slurm
```

Để chạy cascaded benchmark thay vì video bookmarks qua job array:

```bash
EVAL_MODE=cascaded sbatch slurm/eval_uit_array.slurm
```

### Chạy 1 job đơn lẻ (Partition normal)

```bash
sbatch slurm/eval_uit_normal.slurm
```

### Chạy trực tiếp qua môi trường Conda của Lab (Nhanh nhất, không cần Docker/SIF)

Nếu không muốn build/chuyển file container `.tar` lớn qua WinSCP, có thể dùng trực tiếp môi trường Conda `env_ai` có sẵn trên server:

```bash
# 1. Tự động tìm và liên kết các file weights có sẵn trên server
bash scripts/link_server_weights.sh

# 2. Chạy pipeline 9 stage trên video qua Slurm
sbatch slurm/eval_uit_conda.slurm

# Hoặc chạy cascaded benchmark trên tập SoccerNet:
EVAL_MODE=cascaded sbatch slurm/eval_uit_conda.slurm
```

### Chạy debug (Partition small)

```bash
sbatch slurm/eval_uit_small.slurm
```

---

## 5. Theo dõi và kiểm tra kết quả

Kiểm tra trạng thái hàng đợi:

```bash
squeue -u $USER
```

Đọc log thời gian thực:

```bash
# Job array (3 worker)
tail -f /datastore/$USER/slurm_logs/eval_array_*_task_0.out
tail -f /datastore/$USER/slurm_logs/eval_array_*_task_1.out
tail -f /datastore/$USER/slurm_logs/eval_array_*_task_2.out

# Job đơn
tail -f /datastore/$USER/slurm_logs/eval_normal_*.out
```

Hủy job:

```bash
scancel <job_id>
scancel -u $USER
```

---

## 6. Lệnh kiểm tra tài nguyên

| Mục đích | Lệnh |
|---|---|
| Kiểm tra dung lượng thư mục Home | `du -sh /home/$USER` (phải dưới 10 GB) |
| Kiểm tra dung lượng Datastore | `du -sh /datastore/$USER` |
| Thống kê tài nguyên sau khi chạy | `sacct -j <job_id> --format=JobID,JobName,AllocCPUS,Elapsed,MaxRSS,State` |
| Xem chi tiết cấu hình job | `scontrol show job <job_id>` |
