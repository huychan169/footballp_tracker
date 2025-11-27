# Ball → 2D Heat Map Review (hiện trạng)

## Luồng đang chạy
- `Tracker.get_object_tracks` ghi nhận bóng trực tiếp từ YOLO mỗi frame (không track), lưu vào `tracks["ball"][frame_idx][1] = {"bbox": ...}` (`trackers/tracker.py:192-205`).
- Mỗi frame `main.py` gọi `vt.compute_ball` nếu có bbox bóng; hàm này yêu cầu `has_valid_h`, normalizes bbox, rồi unproject xuống sân bằng `CamCalib.calibrate_player_feet` (giả định bóng chạm đất) (`view_transformer/view_transformer.py:95-123`, `FieldMarkings/run.py:107-124`, `main.py:432-436`).
- `vt.update_ball` chỉ push điểm mới vào `Minimap2D.ball_hist` (deque len=3) mà không làm mượt hay TTL (`view_transformer/view_transformer.py:130-132`, `view_transformer/minimap.py:25-78`).
- Khi render, minimap lấy điểm cuối (`ball_hist[-1]`), vẽ viền đen + lõi trắng với bán kính dựa trên `dot_radius` (cùng thông số với player), rồi blend lên khung hình (`view_transformer/minimap.py:120-129`, `main.py:437-439`).

## Vấn đề chính
1) Không có tracking/làm mượt cho bóng: YOLO trả về từng frame và `compute_ball` lấy detection đầu tiên, không clamp vận tốc, không EMA/Kalman; `ball_hist` lưu 3 điểm nhưng render chỉ dùng điểm cuối ⇒ nhấp nháy/teleport khi detection lệch.
2) Bóng bị “kẹt” khi mất detection: `ball_hist` không có TTL hay clear; nếu YOLO mất bóng, minimap giữ nguyên vị trí cũ vô thời hạn cho tới khi có detection khác.
3) Phép chiếu giả định bóng nằm trên mặt cỏ: tái dùng `calibrate_player_feet` (tâm đáy bbox) cho bóng nên mọi cú bổng/pha cản trở đều bị ép xuống mặt sân, dẫn tới sai tọa độ (đặc biệt khi bóng ở trên không hoặc bbox bị bóng đổ/goalpost kéo dài).
4) Chọn detection mang tính tùy ý: lặp qua dict bóng theo thứ tự chèn, không dùng confidence/IoU/knn với vị trí cũ; YOLO trả nhiều bbox hoặc false positive ở bảng quảng cáo có thể bị lấy nhầm. Không có bộ lọc kích thước/tỷ lệ cho bóng nhỏ.
5) Hiển thị chưa tách biệt: bóng dùng cùng `dot_radius` base với player và không có quỹ đạo/tốc độ, nên khó phân biệt khi overlay dày; margin clip làm tròn bất đối xứng (do không mượt).

## Gợi ý cải thiện nhanh
- Track & smooth: chạy ByteTrack/Kalman riêng cho class bóng (hoặc EMA + clamp tốc độ như player) trước khi đẩy vào minimap; dùng `ball_hist` để nội suy và vẽ short tail 3–6 điểm.
- TTL & missing state: reset bóng sau N frame không detection, hoặc mờ dần; khi H xấu hoặc bóng ở trên không, hiển thị trạng thái “lost/air” thay vì giữ vị trí cũ.
- Lọc detection: giữ confidence, chọn bbox cao nhất sau khi lọc size/aspect hợp lý; reject bước nhảy lớn trên sân; ưu tiên bbox gần vị trí trước (gating theo khoảng cách).
- Phép chiếu: với bóng bổng, tạm thời skip chiếu (giữ vị trí trước + flag “air”); khi bóng trên sân, dùng tâm bbox thay vì đáy nếu không có pose/độ cao.
- UI: thu nhỏ radius bóng vs player, dùng viền màu cam/hightlight; thêm tail mờ để dễ đọc hướng di chuyển.

## ByteTrack cho bóng trong pipeline hiện tại
- Cách tích hợp: tái dùng `sv.ByteTrack` (đang có) với một tracker riêng chỉ nhận detections lớp “ball”; chạy online, không cần ReID, giữ một ID cho bóng.
- Ưu điểm: làm mượt và chống nhấp nháy (motion model + association), giữ TTL ngắn khi mất detection, ổn định dù homography tạm xấu (vì chạy trên ảnh), triển khai nhanh chi phí thấp.
- Nhược điểm/lưu ý: dễ drop/đổi ID khi bóng bổng (IoU thấp), cần lọc size/aspect/score trước khi feed để tránh false positive bảng quảng cáo, không tự nhận “air/ground”, phải đồng bộ fps/reset khi tua/skip.
