from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from view_transformer import ViewTransformer2D
from view_transformer.minimap import Minimap2D


def export_ball_trail_video(
    ball_records: List[Dict[str, Any]],
    vt: ViewTransformer2D,
    output_path,
    fps: float,
    trail_frames: int = 200,
):
    """
    Xuất video minimap chỉ hiển thị quỹ đạo bóng (5–7 giây fade dần).
    """
    if not ball_records:
        print("[BallTrail] Không có dữ liệu bóng để xuất.")
        return

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_pitch = vt.base_pitch
    if base_pitch is None:
        base_pitch = np.zeros((vt.cam.IMG_H, vt.cam.IMG_W, 3), dtype=np.uint8)
        base_pitch[:] = (40, 110, 40)

    h, w = base_pitch.shape[:2]

    minimap = Minimap2D(
        width=w,
        height=h,
        dot_radius=vt.minimap.dot_radius,
        ema_alpha=vt.minimap.ema_alpha,
        max_step=vt.minimap.max_step,
        margin=vt.minimap.margin,
        player_ttl=vt.minimap.player_ttl,
        ball_tail=vt.minimap.ball_tail,
        ball_ema_alpha=vt.minimap.ball_ema_alpha,
        ball_max_step=vt.minimap.ball_max_step,
        ball_ttl=max(trail_frames, vt.minimap.ball_ttl),
        ball_tail_long=trail_frames,
    )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 25.0,
        (w, h),
    )

    for idx, rec in enumerate(ball_records):
        canvas = base_pitch.copy()
        minimap.update_ball(
            rec.get("xy"),
            frame_idx=idx,
            is_airball=rec.get("air", False),
            keep_last=rec.get("keep_last", False),
        )
        canvas = minimap.draw_ball_trail(canvas, thickness=2, use_long_history=True)
        ball_xy = minimap.get_last_ball_xy()
        if ball_xy is not None:
            bx, by = map(int, ball_xy)
            border_r = max(minimap.dot_radius - 2, 3)
            inner_r = max(minimap.dot_radius - 4, 2)
            cv2.circle(canvas, (bx, by), border_r, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (bx, by), inner_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"[BallTrail] Xuất video: {out_path} ({len(ball_records)} frames)")
