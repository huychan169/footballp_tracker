from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.append('../')
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
                track_high_thresh=0.6, # giảm 
                track_low_thresh=0.3, # increase in b5
                new_track_thresh=0.7, # tăng
                track_buffer=150, # -> giảm ID switch = tăng track_buffer + giảm match_thresh 
                match_thresh=0.75,
                proximity_thresh=0.5, # khi có reID
                appearance_thresh=0.25, # reID
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

        ## --- add buffer keep lost track ---
        ## Start:
        # self.lost_tracks = {}
        # self.max_lost_time = 360
        # self.prev_active_ids = set()
        # self.last_bbox_per_id = {}
        ## End
    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    def interpolate_ball_positions(self, ball_positions, window=300):
        """
        Nội suy vị trí bóng bằng pandas.interpolate(), nhưng chỉ dùng 
        một đoạn ngắn gần cuối để tránh phình RAM.
        
        Args:
            ball_positions: danh sách dict mỗi frame {1: {'bbox':[x1,y1,x2,y2]}}
            window: số frame tối đa giữ trong bộ đệm (mặc định 300)
        """
        # Giữ tối đa 'window' frame gần nhất
        if len(ball_positions) > window:
            ball_positions = ball_positions[-window:]

        bboxes = [x.get(1, {}).get('bbox', [np.nan]*4) for x in ball_positions]
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate() và backfill để làm đầy NaN
        df = df.interpolate(limit_direction='both')
        df = df.bfill().ffill()

        # Chuyển ngược về dạng [{1:{'bbox':[...]}}]
        out = [{1: {'bbox': row.tolist()}} for _, row in df.iterrows()]
        return out
   

    ## Hàm ktra vị trí xhien 
    ## Start (new)
    # def is_edge_appearance(self, bbox, frame_width, frame_height):
    #     x1, y1, x2, y2 = bbox
    #     center_x = (x1 + x2) / 2
    #     center_y = (y1 + y2) / 2
        
    #     edge_threshold = 80  # 80 pixel từ biên
        
    #     return (center_x < edge_threshold or 
    #             center_x > frame_width - edge_threshold or
    #             center_y < edge_threshold or 
    #             center_y > frame_height - edge_threshold)
    ## End (new)
        

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                    frames[i:i + batch_size],
                    imgsz=960,     # tăng size để bắt bóng nhỏ
                    conf=0.3,     # hạ ngưỡng confidence
                    iou=0.45
                )
            detections += detections_batch
        return detections
    
    ## Add match_lost_tracks()
    ## Start
    # def iou(self, boxA, boxB):
    #     xA = max(boxA[0], boxB[0])
    #     yA = max(boxA[1], boxB[1])
    #     xB = min(boxA[2], boxB[2])
    #     yB = min(boxA[3], boxB[3])

    #     interW = max(0, xB - xA + 1)
    #     interH = max(0, yB - yA + 1)
    #     interArea = interW * interH

    #     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    #     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    #     iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    #     return iou


    # def match_lost_tracks(self, new_bbox):
    #     best_id = None
    #     best_iou = 0

    #     for lost_id, info in self.lost_tracks.items():
    #         if (self.current_frame - info["last_frame"]) > self.max_lost_time:
    #             continue

    #         iou_score = self.iou(new_bbox, info["bbox"])
    #         if iou_score > 6 and iou_score > best_iou:  # ngưỡng có thể chỉnh
    #             best_id = lost_id
    #             best_iou = iou_score

    #     return best_id
    ## End

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "refs": [], "ball": []}

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):
            # Start 
            # self.current_frame = frame_num   # <--- thêm dòng này để match_lost_tracks dùng được
            # frame_height, frame_width = frame.shape[:2]
            # for lost_id in list(self.lost_tracks.keys()):
            #     if (self.current_frame - self.lost_tracks[lost_id]["last_frame"]) > self.max_lost_time:
            #         del self.lost_tracks[lost_id]
            # End

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

                # # Start
                # active_ids = set()
                # # End

                detection_with_tracks = []
                for track in tracks_active:
                    if not track.is_activated:
                        continue

                    tlwh = track.tlwh
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    track_id = track.track_id
                    
                    # Lấy cls_id từ det_input (BoT-SORT không lưu class -> tự map)

                    # # Start (keep 1st line)
                    cls_id = track.cls if hasattr(track, "cls") and track.cls is not None else cls_names_inv["player"]
                    # cls_id = track.cls if hasattr(track, "cls") else cls_names_inv["player"]
                    # # End
                    
                    # # Start (new)
                    # final_track_id = track_id

                    # if track_id not in self.last_bbox_per_id:
                    #     if not self.is_edge_appearance(bbox, frame_width, frame_height):
                    #         # Track xuất hiện giữa khung hình -> ưu tiên thử khôi phục
                    #         recovered_id = self.match_lost_tracks(bbox)
                    #         if recovered_id is not None:
                    #             print(f"RECOVERED: Track {track_id} -> {recovered_id} (middle appearance)")
                    #             final_track_id = recovered_id
                    #             # Xóa khỏi lost buffer
                    #             if recovered_id in self.lost_tracks:
                    #                 del self.lost_tracks[recovered_id]
                    #         else:
                    #             print(f"WARNING: Track {track_id} appeared in middle, no recovery - ACCEPT NEW ID")
                    #     else:
                    #         print(f"NEW: Track {track_id} appeared from edge - ACCEPTED")
                    # # End (new)

                    # # Start (keep 1st line)
                    detection_with_tracks.append((bbox, cls_id, track_id))
                    # detection_with_tracks.append((bbox, cls_id, final_track_id))
                    # active_ids.add(final_track_id)
                    # # End (new)

                    # # Start
                    # # Update lost buffer
                    # if final_track_id in self.lost_tracks:
                    #     del self.lost_tracks[final_track_id]
                    # self.last_bbox_per_id[final_track_id] = bbox
                    # # End

                # # Start
                # # Update lost_tracks
                # # Tìm ID nào vừa mất ở frame này
                # just_lost_ids = self.prev_active_ids - active_ids
                # for lost_id in just_lost_ids:
                #     if lost_id in self.last_bbox_per_id:
                #         self.lost_tracks[lost_id] = {
                #             "bbox": self.last_bbox_per_id[lost_id],
                #             "last_frame": frame_num
                #         }

                # # --- Thử khôi phục ID từ lost buffer --- (new)
                # matched_bboxes = {tuple(track[0]) for track in detection_with_tracks}
                
                # # Kiểm tra các detection chưa được match
                # additional_tracks = []
                # for xyxy, cls_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                #     bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    
                #     # Nếu bbox này chưa được track match
                #     if tuple(bbox) not in matched_bboxes and cls_id == cls_names_inv['player']:
                #         # Thử khôi phục từ lost buffer
                #         recovered_id = self.match_lost_tracks(bbox)
                #         if recovered_id is not None:
                #             # print(f"RECOVERED: Untracked detection -> {recovered_id}")
                #             additional_tracks.append((bbox, cls_id, recovered_id))
                #             # Xóa khỏi lost buffer
                #             if recovered_id in self.lost_tracks:
                #                 del self.lost_tracks[recovered_id]
                #             # Cập nhật bbox
                #             self.last_bbox_per_id[recovered_id] = bbox

                # # Thêm các track được khôi phục
                # detection_with_tracks.extend(additional_tracks)
                
                # # Cập nhật lại active_ids để bao gồm recovered tracks
                # for _, _, recovered_id in additional_tracks:
                #     active_ids.add(recovered_id)
                
                # self.prev_active_ids = active_ids

                # # End
        
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

        cv2.ellipse(
            frame,
            center=(x_center, y2),  
            axes=(int(0.8 * width), int(0.3 * width)),  
            angle=0.0,
            startAngle=-45,
            endAngle=235,  
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

            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

            cv2.putText(
                frame,
                f"{track_id}",
                (x1_rect + 8, y2_rect - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
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

