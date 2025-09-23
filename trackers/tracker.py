from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
from types import SimpleNamespace
import numpy as np

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

# Import BoT-SORT
sys.path.append("BoT-SORT")
from tracker.bot_sort import BoTSORT


class Tracker:
    def __init__(self, model_path, use_boost=False):
        self.model = YOLO(model_path)
        self.use_boost = use_boost

        if self.use_boost:
            args = SimpleNamespace(
                track_high_thresh=0.3,
                track_low_thresh=0.05,
                new_track_thresh=0.4,
                track_buffer=30,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                fast_reid_config=None,
                fast_reid_weights=None,
                device="cuda",
                cmc_method="orb",
                name="exp",
                ablation=False,
                mot20=True   # True nếu có nhiều đối tượng chen lấn
            )
            self.tracker = BoTSORT(args, frame_rate=30)
        else:
            self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "refs": [], "ball": []}

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper → Player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # --- Update tracker ---
            # if self.use_boost:
            #     # chuẩn bị input [x1,y1,x2,y2,score,cls]
            #     det_input = []
            #     for xyxy, score, cls_id in zip(
            #         detection_supervision.xyxy,
            #         detection_supervision.confidence,
            #         detection_supervision.class_id
            #     ):
            #         x1, y1, x2, y2 = xyxy.tolist()
            #         det_input.append([x1, y1, x2, y2, float(score), int(cls_id)])

            #     det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
            #     tracks_active = self.tracker.update(det_input, frame)

            #     detection_with_tracks = []
            #     for track in tracks_active:
            #         if not track.is_activated:
            #             continue
            #         tlwh = track.tlwh
            #         x1, y1, w, h = tlwh
            #         x2, y2 = x1 + w, y1 + h
            #         bbox = [int(x1), int(y1), int(x2), int(y2)]
            #         track_id = track.track_id
            #         cls_id = getattr(track, "cls", cls_names_inv["player"])  # default player
            #         detection_with_tracks.append((bbox, cls_id, track_id))

                # --- Update tracker ---
            if self.use_boost:
                det_input = []
                det_cls_map = {}
                for idx, (xyxy, score, cls_id) in enumerate(zip(
                    detection_supervision.xyxy,
                    detection_supervision.confidence,
                    detection_supervision.class_id
                )):
                    x1, y1, x2, y2 = xyxy.tolist()
                    det_input.append([x1, y1, x2, y2, float(score), int(cls_id)])
                    det_cls_map[idx] = int(cls_id)   # lưu cls gốc theo thứ tự

                det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
                tracks_active = self.tracker.update(det_input, frame)

                detection_with_tracks = []
                for track in tracks_active:
                    if not track.is_activated:
                        continue
                    tlwh = track.tlwh
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    track_id = track.track_id

                    # Lấy cls_id từ det_input (BoT-SORT không lưu class nên ta phải tự map)
                    cls_id = track.cls if hasattr(track, "cls") and track.cls is not None else cls_names_inv["player"]


                    detection_with_tracks.append((bbox, cls_id, track_id))


            else:
                # ByteTrack giữ nguyên
                detection_with_tracks = []
                for det in self.tracker.update_with_detections(detection_supervision):
                    bbox = det[0]
                    cls_id = det[3]
                    track_id = det[4]
                    detection_with_tracks.append((bbox, cls_id, track_id))

            # --- Lưu vào dict ---
            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            for bbox, cls_id, track_id in detection_with_tracks:
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['ref']:
                    tracks["refs"][frame_num][track_id] = {"bbox": bbox}

            # bóng thì lấy trực tiếp từ detection
            for bbox, cls_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox.tolist()}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Vẽ ellipse nhỏ gọn ngay dưới chân
        cv2.ellipse(
            frame,
            center=(x_center, y2),   # dịch ellipse xuống chút cho thoáng
            axes=(int(0.8 * width), int(0.3 * width)),  # ellipse gọn
            angle=0.0,
            startAngle=-45,
            endAngle=235,  # nửa ellipse phía trên
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Hiển thị track_id (nếu có)
        if track_id is not None:
            rectangle_width = 30
            rectangle_height = 16
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + 10
            y2_rect = y1_rect + rectangle_height

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (x1_rect + 8, y2_rect - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return frame

    # --- Vẽ kết quả ---
    def draw_bbox_with_id(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id is not None:
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 40, y1), color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame



    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            ref_dict = tracks["refs"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw Ref
            for _, ref in ref_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_bbox_with_id(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames

