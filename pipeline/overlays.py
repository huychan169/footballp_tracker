from typing import Dict
import cv2


def draw_jersey_labels(frame, players: Dict):
    if not players:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    for pid, info in players.items():
        jersey = info.get('jersey_number')
        bbox = info.get('bbox')
        if not jersey or not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        text = str(jersey)
        org_x = x1
        org_y = max(20, y1 - 8)
        cv2.putText(frame, text, (org_x, org_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_id_mapping_overlay(frame, jersey_display_map: Dict[str, int]):
    if not jersey_display_map:
        return frame
    entries = [f"{jersey} → ID {pid}" for jersey, pid in sorted(jersey_display_map.items(), key=lambda x: str(x[0]))]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    padding = 10
    line_h = 18
    header = "Jersey ↔ Track"
    text_widths = [cv2.getTextSize(text, font, scale, thickness)[0][0] for text in entries + [header]]
    box_w = padding * 2 + max(text_widths + [150])
    box_h = padding * 2 + line_h * (len(entries) + 1)
    h, w = frame.shape[:2]
    x1 = max(0, w - box_w - 20)
    y1 = 20
    x2 = min(w - 10, x1 + box_w)
    y2 = min(h - 10, y1 + box_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, header, (x1 + padding, y1 + padding + 12), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    for idx, text in enumerate(entries):
        y = y1 + padding + line_h * (idx + 1) + 12
        cv2.putText(frame, text, (x1 + padding, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return frame
