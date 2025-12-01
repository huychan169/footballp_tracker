# Homography/view_transformer/view_transformer.py
from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import cv2

from Homography.field_markings.run import blend  # chỉ dùng hàm blend overlay
from .minimap import Minimap2D


class ViewTransformer2D:
    """
    Dùng homography của PnLCalib: H_img2pitch (ảnh → sân, đơn vị mét, gốc giữa sân)
    để:
      - chiếu chân cầu thủ / bóng sang toạ độ sân
      - hiển thị minimap.

    Không phụ thuộc CamCalib nữa.
    """

    def __init__(
        self,
        img_width: int,
        img_height: int,
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
        ball_tail_long: int = 200,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        minimap_width: int = 420,
    ) -> None:
        self.img_w = int(img_width)
        self.img_h = int(img_height)

        self.pitch_length = float(pitch_length)  # 105m
        self.pitch_width = float(pitch_width)    # 68m

        self.scale = float(scale)
        self.alpha = float(alpha)
        self.feet_offset_ratio = float(max(0.0, feet_offset_ratio))

        # tỉ lệ khung minimap ~ tỉ lệ sân
        aspect = self.pitch_width / self.pitch_length  # 68/105 ≈ 0.647
        self.minimap_w = int(minimap_width)
        self.minimap_h = int(round(self.minimap_w * aspect))

        self.minimap = Minimap2D(
            width=self.minimap_w,
            height=self.minimap_h,
            dot_radius=dot_radius,
            ema_alpha=ema_alpha,
            max_step=max_step,
            margin=margin,
            player_ttl=player_ttl,
            ball_tail=ball_tail,
            ball_ema_alpha=ball_ema_alpha,
            ball_max_step=ball_max_step,
            ball_ttl=ball_ttl,
            ball_tail_long=ball_tail_long,
        )

        self.base_pitch: Optional[np.ndarray] = self._make_base_pitch()
        self.H_img2pitch: Optional[np.ndarray] = None
        self.has_valid_h: bool = False

    # ---------- H update từ PnLCalib ----------

    def update_homography(self, H_img2pitch: Optional[np.ndarray]) -> None:
        if H_img2pitch is None:
            self.H_img2pitch = None
            self.has_valid_h = False
        else:
            self.H_img2pitch = H_img2pitch.astype(np.float64)
            # normalise để ổn định
            if self.H_img2pitch[-1, -1] != 0:
                self.H_img2pitch /= self.H_img2pitch[-1, -1]
            self.has_valid_h = True

    # ---------- pitch image ----------

    def _make_base_pitch(self) -> np.ndarray:
        """
        Vẽ sân 2D đơn giản: nền xanh + đường biên trắng + đường giữa + vòng tròn giữa.
        Kích thước: minimap_w x minimap_h.
        """
        pitch = np.zeros((self.minimap_h, self.minimap_w, 3), dtype=np.uint8)
        pitch[:, :] = (0, 120, 0)  # xanh cỏ

        # đường biên
        cv2.rectangle(
            pitch,
            (2, 2),
            (self.minimap_w - 3, self.minimap_h - 3),
            (255, 255, 255),
            2,
        )

        # đường giữa sân
        mid_x = self.minimap_w // 2
        cv2.line(
            pitch,
            (mid_x, 2),
            (mid_x, self.minimap_h - 3),
            (255, 255, 255),
            2,
        )

        # vòng tròn giữa sân
        radius = int(min(self.minimap_w, self.minimap_h) * 0.12)
        cv2.circle(
            pitch,
            (mid_x, self.minimap_h // 2),
            radius,
            (255, 255, 255),
            2,
        )

        return pitch

    # ---------- helper: project 1 điểm ảnh → minimap ----------

    def _project_img_point_to_pitch(
        self, x_img: float, y_img: float
    ) -> Optional[Tuple[float, float]]:
        """
        Dùng H_img2pitch (PnLCalib) để:
          (x_img, y_img) → (sx, sy) trong minimap (pixel).
        PnLCalib dùng hệ toạ độ sân:
          - gốc ở giữa sân
          - trục X dọc chiều dài: [-52.5, 52.5]
          - trục Y dọc chiều ngang: [-34, 34]
        Ta convert:
          world_x [-L/2, L/2]  → [0, minimap_w]
          world_y [-W/2, W/2]  → [minimap_h, 0] (đảo chiều để trên là phía khung thành trên)
        """
        if not self.has_valid_h or self.H_img2pitch is None:
            return None

        p = np.array([x_img, y_img, 1.0], dtype=np.float64)
        wp = self.H_img2pitch @ p
        if abs(wp[2]) < 1e-8:
            return None

        wx = wp[0] / wp[2]
        wy = wp[1] / wp[2]

        # map mét → minimap pixel
        sx = (wx + self.pitch_length / 2.0) / self.pitch_length * self.minimap_w
        sy = (self.pitch_width / 2.0 - wy) / self.pitch_width * self.minimap_h

        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None

        if sx < 0 or sx >= self.minimap_w or sy < 0 or sy >= self.minimap_h:
            # nằm ngoài sân
            return None

        return float(sx), float(sy)

    # ---------- feet / ball projection ----------

    def compute_feet(
        self,
        frame_w: int,
        frame_h: int,
        players: Dict[int, Dict],
    ) -> Dict[int, Tuple[float, float]]:
        res: Dict[int, Tuple[float, float]] = {}
        if not self.has_valid_h or self.H_img2pitch is None:
            return res

        for pid, info in players.items():
            x1, y1, x2, y2 = info["bbox"]
            h = y2 - y1
            w = x2 - x1
            if h <= 1.0 or w <= 1.0:
                continue

            aspect = w / max(h, 1e-6)
            if aspect > 0.9 or aspect < 0.2:
                continue  # box quá bẹt hoặc quá vuông, nghi ngờ

            # offset chân (chỉnh cho đúng)
            fh_ratio = min(1.0, h / float(frame_h))
            adaptive_offset = self.feet_offset_ratio
            if fh_ratio < 0.12:
                adaptive_offset = max(0.04, self.feet_offset_ratio * 0.5)
            elif fh_ratio > 0.35:
                adaptive_offset = min(0.12, self.feet_offset_ratio * 1.5)

            if adaptive_offset > 1e-6 and h > 1.0:
                foot_y = y2 - adaptive_offset * h
                y2 = max(y1, foot_y)

            # toạ độ điểm chân trong ảnh
            x_foot = 0.5 * (x1 + x2)
            y_foot = y2

            mapped = self._project_img_point_to_pitch(x_foot, y_foot)
            if mapped is None:
                continue
            res[pid] = mapped

        return res

    def compute_ball(
        self,
        frame_w: int,
        frame_h: int,
        balls: Dict[int, Dict],
        use_center: bool = False,
    ) -> Optional[Tuple[float, float]]:
        if not self.has_valid_h or self.H_img2pitch is None or not balls:
            return None

        for _, info in balls.items():
            bbox = info.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if use_center:
                cy = 0.5 * (y1 + y2)
                y2 = cy

            x_ball = 0.5 * (x1 + x2)
            y_ball = y2

            mapped = self._project_img_point_to_pitch(x_ball, y_ball)
            if mapped is None:
                continue
            return mapped

        return None

    # ---------- history & minimap ----------

    def update_history(
        self,
        pid2xy: Dict[int, Tuple[float, float]],
        frame_idx: int,
    ) -> None:
        for pid, xy in pid2xy.items():
            self.minimap.update_player(pid, xy, frame_idx)
        self.minimap.prune(frame_idx)

    def update_ball(
        self,
        ball_xy: Optional[Tuple[float, float]],
        frame_idx: int,
        is_airball: bool = False,
        keep_last: bool = False,
    ) -> None:
        self.minimap.update_ball(ball_xy, frame_idx, is_airball=is_airball, keep_last=keep_last)

    def get_last_ball_xy(self) -> Optional[Tuple[float, float]]:
        return self.minimap.get_last_ball_xy()

    def render_minimap(
        self,
        id2color=None,
        id2label=None,
    ) -> Optional[np.ndarray]:
        if self.base_pitch is None:
            return None
        return self.minimap.render(self.base_pitch, id2color=id2color, id2label=id2label)

    def blend_to_frame(
        self,
        frame_bgr: np.ndarray,
        field_img: Optional[np.ndarray],
    ) -> np.ndarray:
        if field_img is None:
            return frame_bgr
        return blend(frame_bgr, field_img, scale=self.scale, alpha=self.alpha)

    def dump_jsonl(self, frame_idx, pid2xy, path, pid2team=None) -> None:
        import json
        with open(path, "a", encoding="utf-8") as f:
            for pid, (x, y) in pid2xy.items():
                rec = {
                    "frame": int(frame_idx),
                    "pid": int(pid),
                    "x": float(x),
                    "y": float(y),
                }
                if pid2team and pid in pid2team:
                    rec["team"] = int(pid2team[pid])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
