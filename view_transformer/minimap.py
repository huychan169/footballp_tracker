# minimap.py
from collections import defaultdict, deque
import numpy as np
import cv2

class Minimap2D:
    """
    Hiển thị duy nhất chấm hiện tại (không vẽ trail).
    Có EMA smoothing + giới hạn bước nhảy để giảm giật.
    """
    def __init__(
        self,
        width, height,
        dot_radius=8,
        ema_alpha=0.2,      # nhỏ hơn -> mượt hơn (0.2~0.4)
        max_step=30,        # px/frame: chặn bước nhảy bất thường
        margin=6            # px: không vẽ sát mép
    ):
        self.W = int(width)
        self.H = int(height)
        self.dot_radius = int(dot_radius)
        self.ema_alpha = float(ema_alpha)
        self.max_step = float(max_step)
        self.margin = int(margin)

        # chỉ cần 1-2 điểm: điểm hiện tại (và điểm trước đó để giới hạn bước)
        self.history = defaultdict(lambda: deque(maxlen=2))

    def _clip(self, p):
        # clip với margin để tránh vẽ “lọt” ra ngoài biên
        x = float(np.clip(p[0], self.margin, self.W - 1 - self.margin))
        y = float(np.clip(p[1], self.margin, self.H - 1 - self.margin))
        return np.array([x, y], dtype=float)

    def update_player(self, pid, pos_xy):
        """pos_xy: (x, y) minimap pixel (float)"""
        if pos_xy is None:
            return
        newp = np.array(pos_xy, dtype=float)
        if not np.all(np.isfinite(newp)):
            return

        if len(self.history[pid]) > 0:
            prev = self.history[pid][-1]
            # 1) giới hạn bước nhảy
            delta = newp - prev
            norm = float(np.linalg.norm(delta))
            if norm > self.max_step and norm > 1e-6:
                delta = delta * (self.max_step / norm)
                newp = prev + delta

            # 2) EMA smoothing
            newp = self.ema_alpha * newp + (1.0 - self.ema_alpha) * prev

        # 3) clip biên
        newp = self._clip(newp)
        self.history[pid].append(newp)

    def render(self, base_pitch_img, id2color=None, id2label=None):
        """
        Chỉ vẽ chấm hiện tại (không trail, không mũi tên).
        id2color: dict pid -> (B,G,R)
        id2label: dict pid -> text (số áo/ID)
        """
        canvas = base_pitch_img.copy()

        for pid, pts in self.history.items():
            if len(pts) == 0:
                continue

            color = (0, 0, 255)
            if id2color and pid in id2color:
                c = id2color[pid]
                color = (int(c[0]), int(c[1]), int(c[2]))

            cx, cy = tuple(np.round(pts[-1]).astype(int))
            if 0 <= cx < self.W and 0 <= cy < self.H:
                cv2.circle(canvas, (cx, cy), self.dot_radius, color, -1, lineType=cv2.LINE_AA)

                if id2label and pid in id2label:
                    cv2.putText(canvas, str(id2label[pid]), (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        return canvas
