# Pretrained models dùng bởi notebook tổng

Các model được tách theo stage để notebook có thể chạy từ trên xuống mà không phụ thuộc đường dẫn weight rải rác trong workspace.

| Stage | File | Nguồn / cách lấy | SHA-256 đã thử nghiệm |
|---|---|---|---|
| 1 | `stage1_pnlcalib/SV_kp`, `SV_lines` | GitHub release PnLCalib v1.0.0; notebook tự tải | kiểm tra trực tiếp trong notebook |
| 2 | `stage2_sst_rtmw/rtmw_l_384x288.onnx` | OpenMMLab; notebook tự tải | `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054` |
| 2 | `stage2_sst_rtmw/model.pth` | bàn giao thủ công từ nguồn tin cậy; legacy pickle | `b1feccc65378e00bd1b5f9c68b67c08a4d17485c845dfc5a3f939bbf554c6308` |
| 4 | `stage4_sam3d/model.ckpt` | gated Hugging Face `facebook/sam-3d-body-dinov3` | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` |
| 4 | `stage4_sam3d/mhr_model.pt` | gated Hugging Face, cùng repository | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |
| 6 | `stage6_ball/yolo-sn-ball-opt.pt` | SoccerNet-v3D release v1.0.0; notebook tự tải | `a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36` |

Stage 3, 5, 7, 8 và 9 không cần pretrained model mới.
