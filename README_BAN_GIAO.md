# Đọc trước khi tiếp nhận dự án

Cập nhật tài liệu: 20/09/2026. Workspace: `D:\NCKH`. Đây là hướng dẫn tiếp nhận hiện trạng, không phải chứng nhận accuracy hay một bản phát hành đã đóng băng.

## Thứ tự đọc

1. Bảng hiện trạng bên dưới để chọn đúng stage và phiên bản.
2. [Danh mục tài liệu](docs_ban_giao/DANH_MUC_TAI_LIEU.md): tách hướng dẫn tham chiếu khỏi lịch sử phiên bản.
   Đọc thêm [đường dẫn và tải dataset](docs_ban_giao/DATASET_PATHS.md): ưu tiên dữ liệu đã có tại `D:\GSR`, `D:\SoccerNet` và các root ngoài workspace.
3. [Checklist bàn giao](docs_ban_giao/CHECKLIST.md): môi trường, model, dữ liệu, kết quả và việc còn thiếu.
4. README, đặc tả và báo cáo mới nhất của stage cần làm; không chọn lệnh chỉ dựa vào tên folder.

## Bản đồ workspace

| Stage / nhiệm vụ | Folder giữ nguyên | Hiện trạng tài liệu / phiên bản |
|---|---|---|
| 1 — Camera và hệ tọa độ sân | `stage_1_camera_v12` | [README](stage_1_camera_v12/README.md): package 0.12.0; phân biệt ground geometry và vertical 3D |
| 2 — Human detection, role, tracking | `stage2_sst_rtmw_v1.3` | [README](stage2_sst_rtmw_v1.3/README.md), [quickstart](stage2_sst_rtmw_v1.3/QUICKSTART_V13.md); metadata 1.3.0 |
| 3 — 2D WholeBody pose QA | `stage3_pose2d_v0.1` | [README](stage3_pose2d_v0.1/README.md), [đặc tả](stage3_pose2d_v0.1/docs/STAGE3_SPEC.md); metadata 0.1.0 |
| 4 — World-grounded body pose | `stage4_metric3d_v0.2.0` | **0.5.1 ground-first**: [báo cáo hiện hành](stage4_metric3d_v0.2.0/V051_GROUND_FIRST_REPORT.md) |
| 5 — Team affiliation / residual role | `stage5_team_affiliation_v0.1.0` | **0.2.1**: [đặc tả cập nhật](stage5_team_affiliation_v0.1.0/STAGE5_V021.md); legacy V0 vẫn mặc định, nhánh mới opt-in |
| 6 — Ball localization / contact | `stage6_ball_localization_v0.4.4` | **runtime 0.5.1**: [hotfix hiện hành](stage6_ball_localization_v0.4.4/V051_PATCH_NOTES.md); metadata đóng gói còn 0.4.4 |

Không thấy folder Stage 7–9 ở cấp gốc trong đợt rà soát này. Không suy ra toàn pipeline đã hoàn thành.

Stage 5 đọc Stage 3, không bắt buộc đi qua Stage 4. Stage 6 contact FOOT ở v0.5.1 không cần Stage 4 anchor; các trường hợp khác phải tuân theo gate riêng trong hotfix. Không coi số thứ tự stage là một chuỗi phụ thuộc tuyến tính bắt buộc.

## Kết quả Stage 4 có thể mở ngay

- [Báo cáo frame 104](stage4_metric3d_v0.2.0/runs/stage4_v051_frame104/sam3d-pitch-refined/stage4_quality_report.json).
- [Ảnh top-down](stage4_metric3d_v0.2.0/runs/stage4_v051_frame104/sam3d-pitch-refined/selected_frame_world_pose_topdown.png).
- [Kết quả 48 tests đã chạy ở lượt triển khai trước](stage4_metric3d_v0.2.0/validation_reports/PYTEST_V051_ALL.xml).
- [Script chạy lại bằng cache](stage4_metric3d_v0.2.0/rerun_frame104_v051.ps1). Script ghi lại folder kết quả v0.5.1; không cần chạy lại SAM3D.

Ảnh thứ 105 tương ứng index 104. Kết quả chỉ có pose cho 10/13 track. Ground coverage 100% là trên 10 track có prior, không phải 13/13. PASS_SANITY không phải metric-GT accuracy. Temporal chưa được đánh giá bằng lần chạy một frame này.

## Các điểm cần lưu ý trước khi chạy

- Máy hiện tại dùng CPU; hướng dẫn CUDA cũ không phải cấu hình của máy này.
- Nhiều quickstart có đường dẫn mẫu từ lần đóng gói trước. Stage 2 hiện có folder `stage2_sst_rtmw_v1.3`, không phải `stage2_sst_rtmw_v1.3.1`.
- Stage 6: `ball_localization/version.py` ghi 0.5.1 nhưng `pyproject.toml` ghi 0.4.4. Đợt dọn tài liệu này không sửa metadata/code; cần thống nhất trước khi phát hành package.
- Các mục “not executed here” trong tài liệu cũ nói về môi trường đóng gói khi đó; không thay thế báo cáo local có ngày/phiên bản mới hơn.
- Không thay đổi hệ tọa độ, selected frame, track ID hoặc camera giữa các phép so sánh mà không ghi provenance.
- Model/dataset có thể có điều khoản truy cập riêng; bàn giao nội bộ không mặc nhiên cho phép tái phân phối.

## Phạm vi dọn tài liệu

Đã xóa 22 Markdown lịch sử rời (changelog, manifest và patch summary cũ), có bản khôi phục `docs_ban_giao/tai_lieu_lich_su_20260920.zip` giữ nguyên đường dẫn. Giữ README, quickstart, spec, QA và báo cáo hiện hành. Không đổi tên workspace, không sửa code/model/data, không chạy inference hoặc benchmark. Các thư mục cache pytest bị từ chối truy cập không được can thiệp.
