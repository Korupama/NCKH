# Dataset ngoài workspace: kiểm tra trước, tải sau

Dataset không bắt buộc nằm trong `D:\NCKH`. Khi bàn giao, chỉ sao chép workspace là chưa đủ để chạy benchmark. Không tự chuyển, xóa hoặc tải đè dữ liệu ở các folder ngoài workspace.

## Các vị trí đã nhìn thấy trên máy (20/09/2026)

| Vị trí | Quan sát | Cách dùng |
|---|---|---|
| `D:\GSR` | Có `train`, `valid`, `test`, `challenge` | Dataset root ưu tiên cho GSR: Stage 2 dùng `--soccernet-root`, Stage 5 dùng `--gsr-root` |
| `D:\SoccerNet` | Có các thư mục league | Nguồn frame/archive SoccerNet cho Stage 6; kiểm tra match/split yêu cầu trước khi tải thêm |
| `D:\datasets\SoccerNetGS` | Có thư mục | Vị trí GSR thay thế; chưa xác minh nội dung hay trùng với `D:\GSR`, không tự gộp |
| `D:\ball3d\data` | Có thư mục | Tài nguyên ball tiềm năng; chưa xác minh cấu trúc hoặc mức đầy đủ, không tự xem là dataset root hợp lệ |
| `D:\NCKH\stage3_pose2d_v0.1\data\3dsp` | Có nhãn train/test đã thấy khi kiểm kê | Bản 3DSP local hiện có; kiểm tra ảnh/nhãn trước khi tải lại |

Đây là kiểm tra tồn tại/cấu trúc cấp đầu, KHÔNG chứng nhận dữ liệu đầy đủ hoặc đúng release. Các lệnh sau là hướng dẫn, chưa được chạy trong đợt dọn tài liệu.

## GSR: dùng dữ liệu đã có

Kiểm tra Stage 5 (chạy trong folder Stage 5):

```powershell
python benchmark_soccernet_gsr.py inspect --gsr-root "D:\GSR" --split valid
```

Trong các lệnh Stage 2, thay root mẫu `D:\Datasets\SoccerNetGS` bằng `D:\GSR` nếu dùng bản này. Stage 2 và Stage 5 cần cùng release/split để so sánh. Chỉ thấy folder `valid` không có nghĩa đã đủ ảnh và `Labels-GameState.json` cho mọi sequence.

Nếu thiếu GSR: lấy bản được cấp quyền từ kênh phát hành SoccerNet-GSR; chỉ tải phần thiếu vào root đã chọn. Xem [nguồn dữ liệu Stage 2](../stage2_sst_rtmw_v1.3/THIRD_PARTY_AND_DATA.md). Tài liệu này không xác minh lại URL tải trên mạng và không cung cấp downloader GSR mới.

## 3DSP: không tải lại mặc định

Chạy trong folder Stage 3 để kiểm tra bản hiện có:

```powershell
python benchmark_stage3.py inspect-3dsp --root "D:\NCKH\stage3_pose2d_v0.1\data\3dsp"
```

Nếu chưa có bản dùng được và muốn lưu bên ngoài workspace, chỉ khi đó dùng helper đã có:

```powershell
python download_open_assets.py 3dsp --output-dir "D:\datasets"
python benchmark_stage3.py inspect-3dsp --root "D:\datasets\3dsp"
```

Helper giải nén vào output directory; kiểm tra cấu trúc thực tế sau giải nén, không tải đè một bản đang dùng. Khi benchmark, truyền `--root` tới bản đã chọn. Không cần copy dataset vào `NCKH`.

## SoccerNet frame archives / ball

Ưu tiên `D:\SoccerNet` hiện có. [OPEN_ASSETS Stage 6](../stage6_ball_localization_v0.4.4/OPEN_ASSETS.md) mô tả helper chọn archive theo CSV và split, cùng cơ chế đăng nhập cho dữ liệu gated. Chỉ dùng downloader nếu các archive cần thiết còn thiếu. CSV release và model checkpoint là tài nguyên riêng, không đồng nhất với root chứa frame.

Không ghi token vào tài liệu/code. Không tải toàn bộ dataset chỉ vì một lệnh báo sai đường dẫn: kiểm tra root, split, release, tên archive và quyền truy cập trước.

## Bàn giao sang máy khác

Ghi bảng ánh xạ đường dẫn cũ → mới, dataset release/split, tình trạng ảnh/nhãn, quyền truy cập và dung lượng cần thiết. Đổi tham số CLI tương ứng; giữ output benchmark trong folder stage để phân biệt dữ liệu nguồn với kết quả sinh ra. Không coi dataset ngoài `NCKH` là file rác khi dọn workspace.
