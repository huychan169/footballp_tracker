# FootballP Tracker

Pipeline phân tích video bóng đá: tracking cầu thủ/bóng, gán đội, OCR số áo, ước lượng chuyển động camera, unproject lên minimap và xuất video (kèm profile hiệu năng và ball trail).

## Cách chạy nhanh

```bash
# Cài uv (nếu chưa có)
pip install uv  # hoặc curl -Ls https://astral.sh/uv/install.sh | bash

# Cài deps theo uv.lock
uv pip sync

# Chạy pipeline
uv run main.py
```

Tham số/đường dẫn mặc định nằm trong `pipeline/config.py` (input/output, model paths, stride OCR, giới hạn thời gian,...). Sửa `PipelineConfig` hoặc viết hàm build config riêng rồi gọi `run_pipeline(config)`.

## Mục đích & chức năng chính

- Nhận diện & theo dõi: YOLOv8 + BoT-SORT (có boost + ReID) -> players, refs, ball.
- Nhận diện số áo: PARSEQ + pose, gán số áo ổn định theo cache/voting, đồng bộ display id với tracker.
- Gán đội: TeamAssigner huấn luyện tự động khi đủ dữ liệu, tô màu theo đội.
- Ước lượng chuyển động camera: CameraMovementEstimator để ổn định overlay.
- Unproject & minimap: ViewTransformer2D dùng homography từ FieldMarkings để chiếu tọa độ chân/bóng lên mặt sân và blend minimap vào frame.
- Xuất kết quả: video chính + video ball trail; ghi profile thời gian từng bước.

## Luồng pipeline hiện tại

1) Mở video, khởi tạo tracker, TeamAssigner, JerseyCoordinator (OCR + cache), CamCalib, ViewTransformer2D, writer.  
2) Lặp từng frame:  
   - Tracking đối tượng.  
   - OCR số áo + ánh xạ display id ổn định (cache TTL, vote, vị trí dự đoán).  
   - Fit model màu áo một lần khi có đủ player.  
   - Cập nhật homography định kỳ.  
   - Gán đội, vẽ bbox/nhãn, overlay chuyển động camera.  
   - Chiếu chân cầu thủ/trọng tài, cập nhật lịch sử minimap.  
   - Chọn bbox bóng hợp lệ, tính vị trí sân, cập nhật tail.  
   - Render minimap, overlay map Jersey ↔ Track, ghi frame.  
   - Dừng nếu vượt `max_runtime_seconds` (mặc định 300s).  
3) Sau vòng lặp: ghi profile hiệu năng, xuất video ball trail, đóng video/writer.

## Thành phần mã chính

- `main.py`: chỉ khởi tạo config và gọi `run_pipeline`.  
- `pipeline/config.py`: `PipelineConfig` (đường dẫn, tham số, flags), `build_default_config()`.  
- `pipeline/pipeline.py`: vòng lặp pipeline chính, profile, xuất ball trail.  
- `pipeline/jersey.py`: JerseyCoordinator (OCR, cache, ánh xạ display id).  
- `pipeline/ball.py`: chọn bbox bóng hợp lệ.  
- `pipeline/overlays.py`: vẽ số áo và bảng map Jersey ↔ Track.  
- Các modules nền: `trackers`, `team_assigner`, `view_transformer`, `camera_movement_estimator`, `FieldMarkings`, `ocr`, `export_ball_trail_video`.

## Tùy chỉnh nhanh

- Đổi video vào/ra, model paths, stride OCR, TTL cache, số giây chạy: chỉnh `PipelineConfig` trong `pipeline/config.py`.  
- Muốn chọn config động: tạo hàm riêng trong `main.py` (ví dụ đọc args/env) rồi truyền vào `run_pipeline(config)`.

## Hướng phát triển tương lai

- CLI tham số hóa (input/output, chọn models, tắt/bật OCR, stride, thời gian).  
- Logging tốt hơn (progress bar, cảnh báo model/homography).  
- Kiểm thử tự động cho JerseyCoordinator/ball selection.  
- Tối ưu hiệu năng (batching OCR, async I/O, profile GPU).  
- UI web nhỏ để xem minimap/ball trail real-time.  
- Hỗ trợ nhiều preset sân/giải đấu, auto-detect field.  
- Thêm Action Spotting (BallAction) khi sẵn sàng.
