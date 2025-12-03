from OldBallAction.src.ball_action import constants
from OldBallAction.src.utils import post_processing
from collections import defaultdict
import numpy as np
import cv2

def load_video_predictions(pred_path: str):
    raw_predictions_path = "data/models/1_raw_predictions.npz"
    raw_predictions_npz = np.load(str(pred_path))
    frame_indexes = raw_predictions_npz["frame_indexes"]
    raw_predictions = raw_predictions_npz["raw_predictions"]
    video_prediction = defaultdict(lambda: np.zeros(constants.num_classes, dtype=np.float32))

    for frame_index, prediction in zip(frame_indexes, raw_predictions):
        video_prediction[frame_index] = prediction

    video_pred_actions = defaultdict(lambda: np.zeros(constants.num_classes, dtype=np.float32))

    for _, cls_index in constants.class2target.items():
        action_frame_indexes, _ = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **constants.postprocess_params
        )

        for frame_index in action_frame_indexes:
            video_pred_actions[frame_index][cls_index] = 1.0

    return video_prediction, video_pred_actions

class BallActionSpot:
    def __init__(self):
        self.predictions = {cls: [] for cls in constants.classes}
        self.pred_actions = {cls: [] for cls in constants.classes}
        self.video_prediction, self.video_pred_actions = load_video_predictions()

        self.team_stats = {
            "team_1": 0,
            "team_2": 0
        }
        self.player_stats = defaultdict(int)

    def visualize_frame(self, frame, frame_idx, posession_team, possession_player):
        prediction = self.video_prediction[frame_idx]
        pred_action = self.video_pred_actions[frame_idx]

        if np.sum(pred_action) > 0:
            pred_cls = constants.target2class[np.argmax(pred_action)]
            prev_prob = prediction[np.argmax(pred_action)]
            prev_action = pred_cls

            # Prepare text
            text = f"{str(prev_action)} ({prev_prob:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = 25, frame.shape[0] - 25  # Bottom left with margin

            # Draw translucent rectangle
            rect_x1, rect_y1 = x - 10, y - text_height - 10
            rect_x2, rect_y2 = x + text_width + 10, y + baseline + 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Put text
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA
            )
        
        return frame