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
        player_ttl=45,
        ball_tail=6,
        ball_ema_alpha=0.45,
        ball_max_step=25.0,
        ball_ttl=30,
        ball_tail_long=200
    ):
        self.W = int(width)
        self.H = int(height)
        self.dot_radius = int(dot_radius)
        self.ema_alpha = float(ema_alpha)
        self.max_step = float(max_step)
        self.margin = int(margin)
        self.player_ttl = int(max(1, player_ttl))
        self.ball_tail = int(max(1, ball_tail))
        self.ball_tail_long = int(max(self.ball_tail, ball_tail_long))
        self.ball_ema_alpha = float(ball_ema_alpha)
        self.ball_max_step = float(ball_max_step)
        self.ball_ttl = int(max(1, ball_ttl))

        # vị trí cầu thủ
        self.history = {}

        # vị trí bóng
        self.ball_hist = deque(maxlen=self.ball_tail)
        self.ball_hist_long = deque(maxlen=self.ball_tail_long)
        self.ball_color = (60, 160, 255)    # cam tươi cho bóng
        self.ball_border = (0, 0, 0)        # viền đen rõ nét
        self.ball_tail_color = (190, 210, 255)  # cam nhạt cho tail
        self.ball_last_seen = None
        self.ball_air = False

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

    def _smooth_ball_step(self, prev, newp):
        # clamp vận tốc và làm mượt cho bóng
        delta = newp - prev
        norm = float(np.linalg.norm(delta))
        if norm > self.ball_max_step and norm > 1e-6:
            delta = delta * (self.ball_max_step / norm)
            newp = prev + delta
        a = self.ball_ema_alpha
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

    def _expire_ball(self, frame_idx):
        if self.ball_last_seen is None:
            return
        if frame_idx - self.ball_last_seen > self.ball_ttl:
            self.ball_hist.clear()
            self.ball_hist_long.clear()
            self.ball_last_seen = None
            self.ball_air = False

    def update_ball(self, pos_xy, frame_idx, is_airball=False, keep_last=False):
        self._expire_ball(frame_idx)
        if pos_xy is None:
            if keep_last and len(self.ball_hist) > 0:
                self.ball_last_seen = frame_idx
                self.ball_air = is_airball
            return

        newp = np.array(pos_xy, dtype=float)
        if not np.all(np.isfinite(newp)):
            return
        newp_short = newp
        if len(self.ball_hist) > 0:
            prev = self.ball_hist[-1]
            newp_short = self._smooth_ball_step(prev, newp_short)

        newp_short = self._clip(newp_short)
        self.ball_hist.append(newp_short)

        newp_long = newp_short
        if len(self.ball_hist_long) > 0:
            prev_long = self.ball_hist_long[-1]
            newp_long = self._smooth_ball_step(prev_long, newp_short)

        newp_long = self._clip(newp_long)
        self.ball_hist_long.append(newp_long)
        self.ball_last_seen = frame_idx
        self.ball_air = is_airball

    def get_last_ball_xy(self):
        if len(self.ball_hist) == 0:
            return None
        last = self.ball_hist[-1]
        return (float(last[0]), float(last[1]))

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

        # ball với tail
        if len(self.ball_hist) > 0:
            pts = [tuple(np.round(p).astype(int)) for p in self.ball_hist]
            tail_color = np.array(self.ball_tail_color, dtype=float)
            total_segments = len(pts) - 1
            for i in range(1, len(pts)):
                p1, p2 = pts[i - 1], pts[i]
                # fade dần: càng gần đuôi càng nhạt
                strength = i / max(1, total_segments)
                fade = 0.25 + 0.75 * strength  # 0.25 -> 1.0
                color = tuple(int(c * fade) for c in tail_color)
                cv2.line(canvas, p1, p2, color, 2, lineType=cv2.LINE_AA)

            bx, by = pts[-1]
            if 0 <= bx < self.W and 0 <= by < self.H:
                border_r = max(self.dot_radius - 2, 3)
                inner_r = max(self.dot_radius - 4, 2)
                cv2.circle(canvas, (bx, by), border_r,
                           self.ball_border, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (bx, by), inner_r,
                           self.ball_color, -1, lineType=cv2.LINE_AA)
                if self.ball_air:
                    cv2.putText(canvas, "AIR", (bx + 6, by - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                tail_color, 1, cv2.LINE_AA)

        return canvas

    def draw_ball_trail(self, canvas, color=None, thickness=2, use_long_history=True):
        """
        Vẽ trail dài 5–7 giây, fade dần về phía đuôi.
        """
        hist = self.ball_hist_long if use_long_history else self.ball_hist
        if len(hist) < 2:
            return canvas

        pts = list(hist)
        n = len(pts)
        color = self.ball_tail_color if color is None else color

        for i in range(1, n):
            x1, y1 = map(int, pts[i - 1])
            x2, y2 = map(int, pts[i])
            alpha = i / float(max(1, n))
            c = (
                int(color[0] * alpha),
                int(color[1] * alpha),
                int(color[2] * alpha)
            )
            cv2.line(canvas, (x1, y1), (x2, y2), c, thickness, cv2.LINE_AA)

        return canvas
