# Checklist tiếp nhận

## 1. Bảo toàn hiện trạng

- Sao lưu workspace và giữ nguyên các đường dẫn tương đối. Không xóa `runs`, `outputs`, `benchmark_results`, `validation_reports` hoặc `backups` trước khi xác định kết quả nào cần giữ.
- Ghi ngày bàn giao, người nhận, mục tiêu tiếp theo. Không đưa token, mật khẩu hoặc file đăng nhập vào gói chia sẻ.
- Danh mục chỉ phân loại tài liệu dự án; `requirements*.txt`, nhãn dataset `gt.txt`, license và README của third party không phải tài liệu thừa.

## 2. Môi trường và tài nguyên

- Đọc [dataset ngoài workspace](DATASET_PATHS.md). `D:\GSR` và `D:\SoccerNet` nằm ngoài gói code; kiểm tra dữ liệu đã có trước khi tải. Bàn giao cả ánh xạ đường dẫn và quyền truy cập, không nhất thiết sao chép dataset vào `NCKH`.

- Ghi nhận Python/package version của từng môi trường. Không coi copy nguyên `.venv` là cách cài đặt có thể tái lập trên máy khác.
- Stage 4 script local đang trỏ đến `D:\anaconda3\envs\stage4-rtmw3d\python.exe`; SAM worker CPU dùng môi trường riêng `D:\NCKH\.venv-sam3d-cpu`. Các đường dẫn này cần sửa khi đổi máy.
- Xác minh model, config, assets và quyền sử dụng từ `third_party`, `weights` và tài liệu provenance của từng stage. Không tải lại nếu tài nguyên đã có và đã xác minh.
- Giữ nguyên cache SAM3D frame104 để tái lập thí nghiệm mà không tốn thời gian inference CPU.

## 3. Kiểm tra đầu vào / đầu ra

- Cùng video, độ phân giải, frame index, camera convention, pitch origin và đơn vị mét.
- Phân biệt raw-distorted pixels với ảnh đã undistort.
- Giữ track ID xuyên Stage 2/3/4/5/6; không ghép nhầm các run khác nhau.
- Đọc quality gate và missing coverage, không chỉ xem lệnh exit thành công.
- Test pass, integration pass và ground-truth accuracy là ba kết luận khác nhau.

## 4. Việc mở cần giao rõ

- Stage 4: metric GT, ảnh có người nhảy/không tiếp đất, camera/foot sai và temporal nhiều frame; chưa freeze accuracy.
- Stage 5: đọc chính sách opt-in và abstention v0.2.1; không bật heuristic dựa trên kết quả VALID để cải thiện số liệu.
- Stage 6: thống nhất metadata package với runtime; contact/metric accuracy chưa được xác thực chỉ từ hotfix và sanity run.
- Stage 2/3: đối chiếu benchmark hiện có trước khi diễn giải các báo cáo đóng gói cũ là kết quả cuối cùng.
- Stage 7–9: xác nhận kế hoạch/phạm vi với người bàn giao; chưa có folder ở gốc được kiểm kê.

## 5. Quy tắc duy trì tài liệu

- README dẫn tới spec/report hiện hành; changelog và QA theo phiên bản là lịch sử.
- Mỗi thí nghiệm mới dùng output directory riêng, lưu config, input/model checksum và runtime provenance khi hỗ trợ.
- Cập nhật [trang bắt đầu](../README_BAN_GIAO.md) và [danh mục](DANH_MUC_TAI_LIEU.md) khi đổi phiên bản.
- Đợt dọn tài liệu 20/09/2026 không chạy lại tests/benchmark; các số liệu được dẫn về báo cáo gốc, không được tuyên bố là lần xác minh mới.
