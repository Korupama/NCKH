# Stage 2 pretrained models

| File | Vai trò | SHA-256 | Cách lấy |
|---|---|---|---|
| `model.pth` | SST/Faster R-CNN, 6 lớp bóng đá | `b1feccc65378e00bd1b5f9c68b67c08a4d17485c845dfc5a3f939bbf554c6308` | Không có URL công khai đã được xác minh trong artifact hiện tại. Lấy từ gói bàn giao tin cậy và chép thủ công vào thư mục này. |
| `rtmw_l_384x288.onnx` | RTMW-L WholeBody-133 | `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054` | Notebook tải ZIP chính thức từ OpenMMLab và tự trích xuất ONNX. |

`model.pth` chứa Python pickle legacy. Chỉ nạp checkpoint đến từ nguồn dự án đáng tin cậy và luôn kiểm tra SHA-256 trước khi inference.

