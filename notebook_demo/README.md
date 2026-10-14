# NCKH 9-stage end-to-end notebook

Thư mục này chứa **một notebook tổng hợp duy nhất** cho toàn bộ pipeline 9 stage. Các stage được bổ sung lần lượt vào cùng notebook theo luồng: chuẩn bị môi trường → tải mã/weights → chọn video và `t0` → Stage 1 → … → Stage 9 → tổng hợp kết quả cuối.

Notebook hiện đã nối đầy đủ **Stage 1 đến Stage 9** trong cùng một file, gồm tải/kiểm tra model, contract checks, visual QA, và đóng gói artifact cuối.

- Notebook tổng: `NCKH_9_Stage_End_to_End.ipynb`
- Video mẫu/dữ liệu vào dùng chung: `inputs/input.mp4`
- Kết quả Stage 1: `outputs/stage1/`
- Kết quả Stage 2: `outputs/stage2/`
- Kết quả Stage 3–9: `outputs/stage3/` … `outputs/stage9/`
- Tổng kết: `outputs/pipeline_summary.json`
- Gói JSON + ảnh QA: `outputs/nckh_9_stage_artifacts.zip`
- PnLCalib weights: `pretrained_models/stage1_pnlcalib/`
- SST + RTMW weights: `pretrained_models/stage2_sst_rtmw/`
- SAM 3D Body weights: `pretrained_models/stage4_sam3d/`
- SoccerNet-v3D YOLO ball weight: `pretrained_models/stage6_ball/`
- Third-party source sau khi chuẩn bị: `third_party/PnLCalib/`

Mở notebook, chọn đúng kernel Python, khai báo `INPUT_VIDEO` và `T0_FRAME_INDEX` (0-based), rồi chạy lần lượt từ trên xuống. Cell cài dependency và benchmark SoccerNet đều có cờ bật/tắt để tránh tải hoặc cài lại ngoài ý muốn.

Phần Stage 1 sử dụng implementation v12.1 tại `../stage_1_camera_v12`. Không sửa các threshold đã freeze trong package này. Stage 4 và Stage 6 có thể dùng Python environment riêng qua `SAM3D_PYTHON` và `STAGE6_PYTHON`.

Kết quả thử với `inputs/input.mp4`, `t0=104` đã chạy xuyên Stage 1–9. Stage 9 trả `DEGRADED/DEFENDER_REFERENCE` vì video mẫu chưa có contact bóng–cầu thủ được xác nhận; đây là fail-closed có lý do, không phải lỗi thực thi.

