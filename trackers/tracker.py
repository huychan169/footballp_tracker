from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import sys
from types import SimpleNamespace

from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Import BoT-SORT
sys.path.append("BoT-SORT")
from tracker.bot_sort import BoTSORT


class Tracker:
    def __init__(self, model_path, use_boost=False):
        self.model = YOLO(model_path)
        self.use_boost = use_boost

        if self.use_boost:
            args = SimpleNamespace(
                track_high_thresh=0.6,
                track_low_thresh=0.3,
                new_track_thresh=0.7,
                track_buffer=150,
                match_thresh=0.75,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                fast_reid_config=None,
                fast_reid_weights=None,
                device="cuda",
                cmc_method="orb",
                name="exp",
                ablation=False,
                mot20=True,
            )
            self.tracker = BoTSORT(args, frame_rate=30)
        else:
            self.tracker = sv.ByteTrack()

    def detect_batch(self, frames_batch):
        return self.model.predict(
            frames_batch,
            imgsz=960,
            conf=0.3,
            iou=0.45,
            stream=False,
        )

    def stream_tracks(self, frame_iter, batch_size=16):
        buffer = []
        start_idx = 0
        for frame in frame_iter:
            buffer.append(frame)
            if len(buffer) == batch_size:
                detections_batch = self.detect_batch(buffer)
                for i, (frm, det) in enumerate(zip(buffer, detections_batch)):
                    tracks_dict = self._update_one_frame(frm, det)
                    yield (start_idx + i, frm, tracks_dict)
                start_idx += len(buffer)
                buffer.clear()

        if buffer:
            detections_batch = self.detect_batch(buffer)
            for i, (frm, det) in enumerate(zip(buffer, detections_batch)):
                tracks_dict = self._update_one_frame(frm, det)
                yield (start_idx + i, frm, tracks_dict)

    def _update_one_frame(self, frame, detection):
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        det_sup = sv.Detections.from_ultralytics(detection)

        for j, cid in enumerate(det_sup.class_id):
            if cls_names[cid] == "goalkeeper":
                det_sup.class_id[j] = cls_names_inv["player"]

        if self.use_boost:
            det_input = []
            for xyxy, score, cid in zip(det_sup.xyxy, det_sup.confidence, det_sup.class_id):
                x1, y1, x2, y2 = xyxy.tolist()
                det_input.append([x1, y1, x2, y2, float(score), int(cid)])
            det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
            tracks_active = self.tracker.update(det_input, frame)

            detection_with_tracks = []
            for trk in tracks_active:
                if not trk.is_activated:
                    continue
                x1, y1, w, h = trk.tlwh
                bbox = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
                cid = getattr(trk, "cls", cls_names_inv["player"])
                detection_with_tracks.append((bbox, cid, trk.track_id))
        else:
            detection_with_tracks = []
            for det in self.tracker.update_with_detections(det_sup):
                bbox, _, _, cid, tid = det
                detection_with_tracks.append((bbox, cid, tid))

        tracks_cur = {"players": {}, "refs": {}, "ball": {}}
        for bbox, cid, tid in detection_with_tracks:
            if cid == cls_names_inv['player']:
                tracks_cur["players"][tid] = {"bbox": bbox}
            elif cid == cls_names_inv['ref']:
                tracks_cur["refs"][tid] = {"bbox": bbox}

        for bbox, cid in zip(det_sup.xyxy, det_sup.class_id):
            if cid == cls_names_inv['ball']:
                tracks_cur["ball"][1] = {"bbox": bbox.tolist()}

        for obj, d in tracks_cur.items():
            for tid, info in d.items():
                bbox = info["bbox"]
                if obj == "ball":
                    pos = get_center_of_bbox(bbox)
                else:
                    pos = get_foot_position(bbox)
                info["position"] = pos

        return tracks_cur

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(0.8 * width), int(0.3 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

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
                cv2.FILLED,
            )

            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

            cv2.putText(
                frame,
                f"{track_id}",
                (x1_rect + 8, y2_rect - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2,
            )

        return frame

    def draw_bbox_with_id(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id is not None:
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 40, y1), color, cv2.FILLED)
            cv2.putText(
                frame,
                f"{track_id}",
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return frame
