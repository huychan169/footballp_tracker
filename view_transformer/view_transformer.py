# view_transformer.py
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

from FieldMarkings.run import CamCalib, blend
from baseline.baseline_cameras import draw_pitch_homography
from .minimap import Minimap2D

class ViewTransformer2D:
    def __init__(
        self,
        cam_calib: CamCalib,
        scale: float = 0.5,
        alpha: float = 0.4,
        dot_radius: int = 8,
        ema_alpha: float = 0.3,
        max_step: float = 30.0,
        margin: int = 6
    ):
        self.cam = cam_calib
        self.cam.H_prev = None
        self.scale = float(scale)
        self.alpha = float(alpha)
        self.minimap = Minimap2D(
            width=self.cam.IMG_W,
            height=self.cam.IMG_H,
            dot_radius=dot_radius,
            ema_alpha=ema_alpha,
            max_step=max_step,
            margin=margin
        )
        self.base_pitch: Optional[np.ndarray] = None

    def update_homography_from_frame(self, annotated_frame: np.ndarray) -> None:
        self.cam(annotated_frame)
        if self.base_pitch is None and self.cam.H is not None:
            self.base_pitch = self._make_base_pitch()

    def _make_base_pitch(self) -> np.ndarray:
        black = np.zeros((self.cam.IMG_H, self.cam.IMG_W, 3), dtype=np.uint8)
        top_view_h = np.array([
            [self.cam.f, 0, self.cam.IMG_W / 2],
            [0, self.cam.f, self.cam.IMG_H / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        return draw_pitch_homography(black, top_view_h)

    def compute_feet(
        self,
        frame_w: int,
        frame_h: int,
        players: Dict[int, Dict]
    ) -> Dict[int, Tuple[float, float]]:
        res: Dict[int, Tuple[float, float]] = {}
        if self.cam.H is None:
            return res

        for pid, info in players.items():
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h

            feet_xy = self.cam.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
            if feet_xy is None:
                continue

            x, y = float(feet_xy[0]), float(feet_xy[1])
            if np.isfinite(x) and np.isfinite(y):
                x = float(np.clip(x, 0.0, self.cam.IMG_W - 1.0))
                y = float(np.clip(y, 0.0, self.cam.IMG_H - 1.0))
                res[pid] = (x, y)
        return res

    def compute_ball(
        self,
        frame_w: int,
        frame_h: int,
        balls: Dict[int, Dict]
    ) -> Optional[Tuple[float, float]]:

        if self.cam.H is None or not balls:
            return None

        for _, info in balls.items():
            bbox = info.get("bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1_n, y1_n, x2_n, y2_n = x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h
            feet_xy = self.cam.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
            if feet_xy is None:
                continue

            x, y = float(feet_xy[0]), float(feet_xy[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            x = float(np.clip(x, 0.0, self.cam.IMG_W - 1.0))
            y = float(np.clip(y, 0.0, self.cam.IMG_H - 1.0))
            return (x, y)

        return None

    def update_history(self, pid2xy: Dict[int, Tuple[float, float]]) -> None:
        for pid, xy in pid2xy.items():
            self.minimap.update_player(pid, xy)

    def update_ball(self, ball_xy: Optional[Tuple[float, float]]) -> None:
        if ball_xy is not None:
            self.minimap.update_ball(ball_xy)

    def render_minimap(
        self,
        id2color=None,
        id2label=None
    ) -> Optional[np.ndarray]:
        if self.base_pitch is None:
            return None
        return self.minimap.render(self.base_pitch, id2color=id2color, id2label=id2label)

    def blend_to_frame(self, frame_bgr: np.ndarray, field_img: Optional[np.ndarray]) -> np.ndarray:
        if field_img is None:
            return frame_bgr
        return blend(frame_bgr, field_img, scale=self.scale, alpha=self.alpha)

    def dump_jsonl(self, frame_idx, pid2xy, path, pid2team=None) -> None:
        import json
        with open(path, "a", encoding="utf-8") as f:
            for pid, (x, y) in pid2xy.items():
                rec = {"frame": int(frame_idx), "pid": int(pid), "x": float(x), "y": float(y)}
                if pid2team and pid in pid2team:
                    rec["team"] = int(pid2team[pid])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
