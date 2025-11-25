from typing import Dict, Optional


def select_ball_detection(ball_tracks: Dict, frame_w: int, frame_h: int) -> Optional[dict]:
    """Select the most plausible ball bbox."""
    best = None
    best_score = -1.0
    max_area = 0.01 * frame_w * frame_h  # avoid large false positives
    for _, info in ball_tracks.items():
        bbox = info.get("bbox")
        if bbox is None:
            continue
        score = float(info.get("score", 0.0))
        x1, y1, x2, y2 = bbox
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        if w <= 1.0 or h <= 1.0:
            continue
        area = w * h
        if area < 4.0 or area > max_area:
            continue
        aspect = w / max(h, 1e-6)
        if aspect < 0.5 or aspect > 2.5:
            continue
        if score > best_score:
            best_score = score
            best = {"bbox": bbox, "score": score, "area": area}
    return best
