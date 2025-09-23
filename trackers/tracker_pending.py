# from ultralytics import YOLO
# import supervision as sv
# import pickle
# import os
# import sys
# import cv2
# sys.path.append('../')
# from utils import get_center_of_bbox, get_bbox_width

# class Tracker:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path) 
#         self.tracker = sv.ByteTrack()

#     def detect_frames(self, frames):
#         batch_size=20 
#         detections = [] 
#         for i in range(0,len(frames),batch_size):
#             detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
#             detections += detections_batch
#         return detections
    
#     def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

#         if read_from_stub and stub_path is not None and os.path.exists(stub_path):
#             with open(stub_path,'rb') as f:
#                 tracks = pickle.load(f)
#             return tracks

#         detections = self.detect_frames(frames)

#         tracks={
#             "players":[],
#             "refs":[],
#             "ball":[]
#         }

#         for frame_num, detection in enumerate(detections):
#             cls_names = detection.names
#             cls_names_inv = {v:k for k,v in cls_names.items()}

#             # Covert to supervision Detection format
#             detection_supervision = sv.Detections.from_ultralytics(detection)

#             # Convert GoalKeeper to player object
#             for object_ind , class_id in enumerate(detection_supervision.class_id):
#                 if cls_names[class_id] == "goalkeeper":
#                     detection_supervision.class_id[object_ind] = cls_names_inv["player"]

#             # Track Objects
#             detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

#             tracks["players"].append({})
#             tracks["refs"].append({})
#             tracks["ball"].append({})

#             for frame_detection in detection_with_tracks:
#                 bbox = frame_detection[0].tolist()
#                 cls_id = frame_detection[3]
#                 track_id = frame_detection[4]

#                 if cls_id == cls_names_inv['player']:
#                     tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
#                 if cls_id == cls_names_inv['ref']:
#                     tracks["refs"][frame_num][track_id] = {"bbox":bbox}

#             for frame_detection in detection_supervision:
#                 bbox = frame_detection[0].tolist()
#                 cls_id = frame_detection[3]

#                 if cls_id == cls_names_inv['ball']:
#                     tracks["ball"][frame_num][1] = {"bbox":bbox}

#         if stub_path is not None:
#             with open(stub_path,'wb') as f:
#                 pickle.dump(tracks,f)

#         return tracks
#             # break

#     def draw_ellipse(self,frame,bbox,color,track_id=None):
#         y2 = int(bbox[3])
#         x_center, _ = get_center_of_bbox(bbox)
#         width = get_bbox_width(bbox)

#         cv2.ellipse(
#             frame,
#             center=(x_center,y2),
#             axes=(int(width), int(0.35*width)),
#             angle=0.0,
#             startAngle=-45,
#             endAngle=235,
#             color = color,
#             thickness=2,
#             lineType=cv2.LINE_4
#         )

#         rectangle_width = 40
#         rectangle_height=20
#         x1_rect = x_center - rectangle_width//2
#         x2_rect = x_center + rectangle_width//2
#         y1_rect = (y2- rectangle_height//2) +15
#         y2_rect = (y2+ rectangle_height//2) +15

#         if track_id is not None:
#             cv2.rectangle(frame,
#                           (int(x1_rect),int(y1_rect) ),
#                           (int(x2_rect),int(y2_rect)),
#                           color,
#                           cv2.FILLED)
            
#             x1_text = x1_rect+12
#             if track_id > 99:
#                 x1_text -=10
            
#             cv2.putText(
#                 frame,
#                 f"{track_id}",
#                 (int(x1_text),int(y1_rect+15)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0,0,0),
#                 2
#             )

#         return frame

#     def draw_annotations(self,video_frames, tracks):
#         output_video_frames= []
#         for frame_num, frame in enumerate(video_frames):
#             frame = frame.copy()

#             player_dict = tracks["players"][frame_num]
#             ball_dict = tracks["ball"][frame_num]
#             ref_dict = tracks["refs"][frame_num]

#             # Draw Players
#             for track_id, player in player_dict.items():
#                 color = player.get("team_color",(0,0,255))
#                 frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

#             # Draw Ref
#             for _, ref in ref_dict.items():
#                 frame = self.draw_ellipse(frame, ref["bbox"],(0,255,255))

#             # # Draw ball 
#             # for track_id, ball in ball_dict.items():
#             #     frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

#             output_video_frames.append(frame)

#         return output_video_frames



## Keep bbox
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "refs": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)


            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['ref']:
                    tracks["refs"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_bbox_with_id(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)

        # Vẽ rectangle (bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if track_id is not None:
            # Vẽ ô nhỏ để hiển thị track_id
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 40, y1), color, cv2.FILLED)
            cv2.putText(
                frame,
                f"{track_id}",
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
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
                frame = self.draw_bbox_with_id(frame, player["bbox"], color, track_id)

            # Draw Ref
            for track_id, ref in ref_dict.items():
                frame = self.draw_bbox_with_id(frame, ref["bbox"], (0, 255, 255))

            # Draw Ball (nếu muốn bật tracking bóng)
            for track_id, ball in ball_dict.items():
                frame = self.draw_bbox_with_id(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames
            