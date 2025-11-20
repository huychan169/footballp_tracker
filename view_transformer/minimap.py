# view_transformer/minimap.py
from collections import deque
import numpy as np
import cv2

class Minimap2D:

    def __init__(
        self,
        width, height,
        dot_radius=8,
        ema_alpha=0.1,      # nhỏ hơn -> mượt hơn (0.2~0.4)
        max_step=12,        # px/frame: chặn bước nhảy bất thường
        margin=6,
        player_ttl=45
    ):
        self.W = int(width)
        self.H = int(height)
        self.dot_radius = int(dot_radius)
        self.ema_alpha = float(ema_alpha)
        self.max_step = float(max_step)
        self.margin = int(margin)
        self.player_ttl = int(max(1, player_ttl))

        # vị trí cầu thủ
        self.history = {}

        # vị trí bóng
        self.ball_hist = deque(maxlen=3)
        self.ball_color = (255, 255, 255)  # trắng
        self.ball_border = (0, 0, 0)       # viền đen

    def _clip(self, p):
        # clip với margin để tránh vẽ “lọt” ra ngoài biên
        x = float(np.clip(p[0], self.margin, self.W - 1 - self.margin))
        y = float(np.clip(p[1], self.margin, self.H - 1 - self.margin))
        return np.array([x, y], dtype=float)

    def _smooth_step(self, prev, newp):
        # giới hạn bước + EMA
        delta = newp - prev
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step and norm > 1e-6:
            delta = delta * (self.max_step / norm)
            newp = prev + delta
        # new = a*new + (1-a)*prev  (a nhỏ -> bám prev nhiều hơn)
        a = self.ema_alpha
        newp = a * newp + (1.0 - a) * prev
        return newp

    def update_player(self, pid, pos_xy, frame_idx):
        if pos_xy is None:
            return
        newp = np.array(pos_xy, dtype=float)
        if not np.all(np.isfinite(newp)):
            return

        if pid in self.history:
            prev = self.history[pid]["pos"]
            newp = self._smooth_step(prev, newp)

        newp = self._clip(newp)
        self.history[pid] = {"pos": newp, "last_seen": int(frame_idx)}

    def prune(self, frame_idx):
        stale = [pid for pid, meta in self.history.items()
                 if frame_idx - meta.get("last_seen", frame_idx) > self.player_ttl]
        for pid in stale:
            self.history.pop(pid, None)

    def update_ball(self, pos_xy):
        if pos_xy is None:
            return
        newp = np.array(pos_xy, dtype=float)
        if not np.all(np.isfinite(newp)):
            return
        newp = self._clip(newp)
        self.ball_hist.append(newp)

    def _boost_team_color(self, bgr):
        c = np.array([[bgr]], dtype=np.uint8)  # shape (1,1,3)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]

        # nếu là màu xám (saturation thấp) thì chỉ tăng sáng để giữ tông xám
        if s < 30 or (max(bgr) - min(bgr)) < 10:
            s = min(s, 30)
            v = max(v, 215)
        else:
            # màu đậm -> tăng saturation + brightness để dễ nhìn
            s = max(s, 150)
            v = max(v, 200)

        hsv[0, 0] = (h, s, v)
        bgr_boost = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr_boost[0]), int(bgr_boost[1]), int(bgr_boost[2]))

    def render(self, base_pitch_img, id2color=None, id2label=None):

        canvas = base_pitch_img.copy()

        # player
        for pid, meta in self.history.items():
            pos = meta.get("pos")
            if pos is None:
                continue

            color = (0, 0, 255)  # default
            if id2color and pid in id2color:
                color = self._boost_team_color(id2color[pid])

            cx, cy = tuple(np.round(pos).astype(int))
            if 0 <= cx < self.W and 0 <= cy < self.H:
                cv2.circle(canvas, (cx, cy), self.dot_radius, color, -1, lineType=cv2.LINE_AA)

                if id2label and pid in id2label:
                    cv2.putText(canvas, str(id2label[pid]), (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # ball
        if len(self.ball_hist) > 0:
            bx, by = tuple(np.round(self.ball_hist[-1]).astype(int))
            if 0 <= bx < self.W and 0 <= by < self.H:
                # viền
                cv2.circle(canvas, (bx, by), max(self.dot_radius - 2, 3),
                           self.ball_border, -1, lineType=cv2.LINE_AA)
                # bóng
                cv2.circle(canvas, (bx, by), max(self.dot_radius - 3, 2),
                           self.ball_color, -1, lineType=cv2.LINE_AA)

        return canvas
