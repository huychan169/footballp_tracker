# view_transformer.py
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

from Homography.field_markings.run import CamCalib, blend
from Homography.field_markings.baseline_cameras import draw_pitch_homography
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
        margin: int = 6,
        feet_offset_ratio: float = 0.08,
        player_ttl: int = 45,
        ball_tail: int = 6,
        ball_ema_alpha: float = 0.45,
        ball_max_step: float = 25.0,
        ball_ttl: int = 30,
        ball_tail_long: int = 200
    ):
        self.cam = cam_calib
        self.cam.H_prev = None
        self.scale = float(scale)
        self.alpha = float(alpha)
        self.feet_offset_ratio = float(max(0.0, feet_offset_ratio))
        self.minimap = Minimap2D(
            width=self.cam.IMG_W,
            height=self.cam.IMG_H,
            dot_radius=dot_radius,
            ema_alpha=ema_alpha,
            max_step=max_step,
            margin=margin,
            player_ttl=player_ttl,
            ball_tail=ball_tail,
            ball_ema_alpha=ball_ema_alpha,
            ball_max_step=ball_max_step,
            ball_ttl=ball_ttl,
            ball_tail_long=ball_tail_long
        )
        self.base_pitch: Optional[np.ndarray] = None

    def update_homography_from_frame(self, annotated_frame: np.ndarray) -> None:
        self.cam(annotated_frame)
        if self.base_pitch is None and self.cam.H is not None and self.cam.has_valid_h:
            self.base_pitch = self._make_base_pitch()

    def _make_base_pitch(self) -> np.ndarray:
        black = np.zeros((self.cam.IMG_H, self.cam.IMG_W, 3), dtype=np.uint8)
        top_view_h = np.array([
            [self.cam.f, 0, self.cam.IMG_W / 2],
            [0, self.cam.f, self.cam.IMG_H / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        return draw_pitch_homography(black, top_view_h)
    
    def compute_feet_uncalibrated(
        self,
        frame_w: int,
        frame_h: int,
        players: Dict[int, Dict],
        enable_homography: bool = True
    ) -> Dict[int, Tuple[float, float]]:
        
        player_feets = {}
        for pid, info in players.items():
            x1, y1, x2, y2 = info["bbox"]
            height = y2 - y1
            width = x2 - x1
            if height <= 1.0 or width <= 1.0:
                continue
            aspect = width / max(height, 1e-6)

            if aspect > 0.9 or aspect < 0.2:
                continue

            fh_ratio = min(1.0, height / float(frame_h))
            adaptive_offset = self.feet_offset_ratio
            if fh_ratio < 0.12:
                adaptive_offset = max(0.04, self.feet_offset_ratio * 0.5)
            elif fh_ratio > 0.35:
                adaptive_offset = min(0.12, self.feet_offset_ratio * 1.5)
            if self.feet_offset_ratio > 1e-6 and height > 1.0:
                foot_y = y2 - adaptive_offset * height
                y2 = max(y1, foot_y)

            if enable_homography:
                player_feets[pid] = (x1, y1, x2, y2)
            else:
                x_feet = 0.5 * (x1 + x2)
                player_feets[pid] = (int(x_feet), int(y2))
        
        return player_feets

    def compute_feet(
        self,
        frame_w: int,
        frame_h: int,
        players: Dict[int, Dict]
    ) -> Dict[int, Tuple[float, float]]:
        res = {}

        if self.cam.H is None or not getattr(self.cam, "has_valid_h", False):
            return res
        
        feet_uncalibrated = self.compute_feet_uncalibrated(frame_w, frame_h, players)

        for pid, info in feet_uncalibrated.items():
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h
            feet_xy = self.cam.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
            x, y = float(feet_xy[0]), float(feet_xy[1])

            if feet_xy is None:
                continue
            elif np.isfinite(x) and np.isfinite(y):
                x = float(np.clip(x, 0.0, self.cam.IMG_W - 1.0))
                y = float(np.clip(y, 0.0, self.cam.IMG_H - 1.0))
                res[pid] = (x, y)

        return res
    
    def compute_ball_uncalibrated(
        self,
        frame_w: int,
        frame_h: int,
        balls: Dict,
        use_center: bool = False,
        enable_homography: bool = True
    ):
        
        bbox = balls.get("bbox", None)
        
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox

        if use_center:
            cy = 0.5 * (y1 + y2)
            y2 = cy

        if enable_homography:
            return (x1, y1, x2, y2)
        else:
            return (int(0.5 * (x1 + x2)), int(y2))

    def compute_ball(
        self,
        frame_w: int,
        frame_h: int,
        balls: Dict,
        use_center: bool = False
    ) -> Optional[Tuple[float, float]]:

        if self.cam.H is None or not getattr(self.cam, "has_valid_h", False) or not balls:
            return None
        
        ball_uncalibrated = self.compute_ball_uncalibrated(frame_w, frame_h, balls, use_center=use_center)

        x1, y1, x2, y2 = ball_uncalibrated
        x1_n, y1_n, x2_n, y2_n = x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h
        feet_xy = self.cam.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
        if feet_xy is None:
            return None

        x, y = float(feet_xy[0]), float(feet_xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None

        x = float(np.clip(x, 0.0, self.cam.IMG_W - 1.0))
        y = float(np.clip(y, 0.0, self.cam.IMG_H - 1.0))
        
        return (x, y)

    def update_history(self, pid2xy: Dict[int, Tuple[float, float]], frame_idx: int) -> None:
        for pid, xy in pid2xy.items():
            self.minimap.update_player(pid, xy, frame_idx)
        self.minimap.prune(frame_idx)

    def update_ball(self, ball_xy: Optional[Tuple[float, float]], frame_idx: int, is_airball: bool = False, keep_last: bool = False) -> None:
        self.minimap.update_ball(ball_xy, frame_idx, is_airball=is_airball, keep_last=keep_last)

    def get_last_ball_xy(self) -> Optional[Tuple[float, float]]:
        return self.minimap.get_last_ball_xy()

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
