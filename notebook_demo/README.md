# NCKH 9-stage end-to-end notebook

Thư mục này chứa **một notebook tổng hợp duy nhất** cho toàn bộ pipeline 9 stage. Các stage được bổ sung lần lượt vào cùng notebook theo luồng: chuẩn bị môi trường → tải mã/weights → chọn video và `t0` → Stage 1 → … → Stage 9 → tổng hợp kết quả cuối.

Hiện tại notebook đã triển khai hoàn chỉnh **Stage 1 — Camera calibration** và **Stage 2 — Human entities, roles & target-window tracking**. Stage 3–9 sẽ được nối tiếp trong chính file này.

- Notebook tổng: `NCKH_9_Stage_End_to_End.ipynb`
- Video mẫu/dữ liệu vào dùng chung: `inputs/input.mp4`
- Kết quả Stage 1: `outputs/stage1/`
- Kết quả Stage 2: `outputs/stage2/`
- PnLCalib weights: `pretrained_models/stage1_pnlcalib/`
- SST + RTMW weights: `pretrained_models/stage2_sst_rtmw/`
- Third-party source sau khi chuẩn bị: `third_party/PnLCalib/`

Mở notebook, chọn đúng kernel Python, khai báo `INPUT_VIDEO` và `T0_FRAME_INDEX` (0-based), rồi chạy lần lượt từ trên xuống. Cell cài dependency và benchmark SoccerNet đều có cờ bật/tắt để tránh tải hoặc cài lại ngoài ý muốn.

Phần Stage 1 sử dụng implementation v12.1 tại `../stage_1_camera_v12`. Không sửa các threshold đã freeze trong package này.

