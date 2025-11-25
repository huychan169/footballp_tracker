from pathlib import Path
import math
from collections import deque
from typing import Deque, Dict, Set, Tuple
import cv2

from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from view_transformer import ViewTransformer2D
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from camera_movement_estimator import CameraMovementEstimator

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend

from BallAction.Test_Visual import BallActionSpot
from ocr.jersey_recognizer import JerseyRecogniser

PARSEQ_CKPT_PATH = Path("outputs/parseq/03_11_2025/checkpoints/test.ckpt")
POSE_MODEL_PATH = Path("models/yolov8m-pose.pt")
OCR_FRAME_STRIDE = 2
JERSEY_CACHE_TTL_FRAMES = 150
JERSEY_CACHE_MIN_CONFIDENCE = 0.58


def draw_jersey_labels(frame, players):
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


def draw_id_mapping_overlay(frame, jersey_display_map):
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



def main():
    in_path  = 'video/input_videos/match1_cut45s.mp4'
    out_path = 'video/output_videos/match1_cut45s_1117v.avi'

    cap, fps, w, h = open_video(in_path)
    tracker = Tracker(
        'models/best_ylv8_ep50.pt',
        use_boost=True,
        use_reid=True,
        reid_backend='torchreid',
        torchreid_weights='reid/models/model.pth.tar-30'
    )
    team_assigner = TeamAssigner(flip_guard_frames=8)
    try:
        tracker_device = getattr(tracker, "device", "cuda:0")
        jersey_device = "cuda" if str(tracker_device).startswith("cuda") else "cpu"
        jersey_ocr = JerseyRecogniser(
            parseq_root=Path("parseq"),
            checkpoint_path=PARSEQ_CKPT_PATH,
            pose_model_path=POSE_MODEL_PATH,
            device=jersey_device,
            history_window=30,
            confidence_threshold=0.6,
            vote_min_confidence=0.55,
            vote_min_support=2,
            vote_high_threshold=0.65,
            vote_count_min=4,
            vote_count_margin=2,
        )
    except Exception as exc:
        print(f"[JerseyOCR] Không thể khởi tạo OCR: {exc}")
        jersey_ocr = None

    jersey_display_map: Dict[str, int] = {}
    active_jersey_ids: Set[int] = set()
    jersey_position_history: Dict[str, Deque[Tuple[int, float, float]]] = {}
    track_jersey_cache: Dict[int, Dict[str, float]] = {}
    jersey_track_cache: Dict[str, Dict[str, int]] = {}

    writer = create_writer(out_path, fps, w, h)
    team_fit_done = False
    cm_est = None
    frame_idx = 0

    cam_calib = CamCalib(MODEL_PATH, LINES_FILE)
    vt = ViewTransformer2D(
        cam_calib,
        scale=0.5,
        alpha=0.4,
        dot_radius=8,
        ema_alpha=0.2,
        max_step=15,
        margin=6
    )

    ball_action = BallActionSpot()

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        if jersey_ocr is not None:
            if cur_tracks['players']:
                player_entries = list(cur_tracks['players'].items())
                jersey_candidates: Dict[str, list] = {}
                pending_players = []
                for display_id, info in player_entries:
                    bbox = info.get('bbox')
                    source_tid = info.get('source_track_id')
                    decision = None
                    center = None
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                        attempt = (
                            (OCR_FRAME_STRIDE <= 1)
                            or (frame_idx % OCR_FRAME_STRIDE == 0)
                            or ('jersey_number' not in info)
                        )
                        if attempt:
                            reading = jersey_ocr.read_number(frame, (x1, y1, x2, y2), frame_idx=frame_idx)
                            decision = jersey_ocr.confirm_number(display_id, reading, frame_idx=frame_idx)
                        else:
                            decision = jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                    else:
                        decision = jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                    info.pop('jersey_number', None)
                    info.pop('jersey_confidence', None)

                    resolved = None
                    if (
                        decision
                        and decision.is_confirmed
                        and decision.text
                        and (decision.mean_confidence or 0.0) >= JERSEY_CACHE_MIN_CONFIDENCE
                    ):
                        resolved = {
                            "jersey": decision.text,
                            "confidence": decision.mean_confidence,
                            "consensus": decision.consensus,
                            "votes": decision.votes,
                            "source": "ocr",
                        }
                    elif source_tid is not None:
                        cached = track_jersey_cache.get(source_tid)
                        if cached:
                            age = frame_idx - cached.get("frame", -1)
                            jersey = cached.get("jersey")
                            owner = jersey_track_cache.get(jersey) if jersey else None
                            owner_active = (
                                owner
                                and owner.get("track_id") is not None
                                and owner["track_id"] != source_tid
                                and frame_idx - owner.get("frame", -1) <= JERSEY_CACHE_TTL_FRAMES
                            )
                            if (
                                jersey
                                and age <= JERSEY_CACHE_TTL_FRAMES
                                and not owner_active
                            ):
                                resolved = {
                                    "jersey": jersey,
                                    "confidence": cached.get("confidence", 0.5),
                                    "consensus": cached.get("consensus", 0.0),
                                    "votes": cached.get("votes", 0),
                                    "source": "cache",
                                }

                    if resolved:
                        jersey = resolved["jersey"]
                        jersey_candidates.setdefault(jersey, []).append({
                            "info": info,
                            "confidence": resolved["confidence"],
                            "consensus": resolved.get("consensus", 0.0),
                            "votes": resolved.get("votes", 0),
                            "center": center,
                            "source_tid": source_tid,
                            "original_id": display_id,
                            "jersey": jersey,
                        })
                        if source_tid is not None:
                            track_jersey_cache[source_tid] = {
                                "jersey": jersey,
                                "frame": frame_idx,
                                "confidence": resolved["confidence"],
                                "consensus": resolved.get("consensus", 0.0),
                                "votes": resolved.get("votes", 0),
                            }
                            jersey_track_cache[jersey] = {
                                "track_id": source_tid,
                                "frame": frame_idx,
                            }
                    else:
                        pending_players.append((info, source_tid, display_id))

                selected_candidates = []
                for jersey, candidates in jersey_candidates.items():
                    history = jersey_position_history.get(jersey)
                    expected_pos = None
                    loyalty_display = jersey_display_map.get(jersey)
                    if history:
                        last_frame, last_cx, last_cy = history[-1]
                        if frame_idx - last_frame <= 20:
                            expected_pos = (last_cx, last_cy)
                            if len(history) >= 2:
                                prev_frame, prev_cx, prev_cy = history[-2]
                                frame_delta = max(1, last_frame - prev_frame)
                                vx = (last_cx - prev_cx) / frame_delta
                                vy = (last_cy - prev_cy) / frame_delta
                                delta_frames = max(0, frame_idx - last_frame)
                                expected_pos = (last_cx + vx * delta_frames, last_cy + vy * delta_frames)

                    def candidate_key(candidate):
                        center = candidate['center']
                        loyalty_rank = 0 if loyalty_display == candidate['original_id'] else 1
                        if center is None:
                            distance = 1e9
                        else:
                            if expected_pos is not None:
                                distance = math.hypot(center[0] - expected_pos[0], center[1] - expected_pos[1])
                            elif history:
                                _, last_cx, last_cy = history[-1]
                                distance = math.hypot(center[0] - last_cx, center[1] - last_cy)
                            else:
                                distance = 1e9
                        return (
                            loyalty_rank,
                            distance,
                            -candidate['consensus'],
                            -candidate['votes'],
                            -candidate['confidence'],
                        )

                    best_candidate = min(candidates, key=candidate_key)
                    best_info = best_candidate['info']
                    best_info['jersey_number'] = best_candidate['jersey']
                    best_info['jersey_confidence'] = best_candidate['confidence']
                    selected_candidates.append(best_candidate)
                    for other in candidates:
                        if other is best_candidate:
                            continue
                        pending_players.append((other['info'], other['source_tid'], other['original_id']))

                jersey_confirmations = [
                    (cand['jersey'], cand['info'], cand['source_tid'], cand['original_id'])
                    for cand in selected_candidates
                ]

                assigned_players = {}
                new_player_id_map = {}
                used_ids = set()
                if tracker.max_player_ids:
                    all_ids = list(range(1, tracker.max_player_ids + 1))
                else:
                    all_ids = sorted({pid for pid, _ in player_entries})
                reserved_ids = set(jersey_display_map.values())
                display_to_jersey_local = {display_id: jersey for jersey, display_id in jersey_display_map.items()}

                for jersey, info, source_tid, original_id in jersey_confirmations:
                    if jersey not in jersey_display_map:
                        available_for_new = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                        if not available_for_new:
                            continue
                        assigned_display = available_for_new[0]
                        jersey_display_map[jersey] = assigned_display
                        reserved_ids.add(assigned_display)
                    else:
                        assigned_display = jersey_display_map[jersey]
                    if assigned_display in assigned_players:
                        alternatives = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                        if alternatives:
                            assigned_display = alternatives[0]
                            jersey_display_map[jersey] = assigned_display
                            reserved_ids.add(assigned_display)
                        else:
                            continue
                    previous_jersey = display_to_jersey_local.get(assigned_display)
                    if previous_jersey and previous_jersey != jersey:
                        jersey_display_map.pop(previous_jersey, None)
                        display_to_jersey_local.pop(assigned_display, None)
                    display_to_jersey_local[assigned_display] = jersey
                    used_ids.add(assigned_display)
                    info['display_id'] = assigned_display
                    if source_tid is not None:
                        new_player_id_map[source_tid] = assigned_display
                    assigned_players[assigned_display] = info

                temp_pool = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                temp_pool.sort(reverse=True)
                for info, source_tid, original_id in pending_players:
                    if original_id is not None and original_id not in used_ids:
                        assigned_display = original_id
                    elif temp_pool:
                        assigned_display = temp_pool.pop(0)
                    else:
                        remaining = [i for i in all_ids if i not in used_ids]
                        if not remaining:
                            continue
                        assigned_display = remaining[0]
                    used_ids.add(assigned_display)
                    info['display_id'] = assigned_display
                    if source_tid is not None:
                        new_player_id_map[source_tid] = assigned_display
                    assigned_players[assigned_display] = info

                cur_tracks['players'] = dict(sorted(assigned_players.items()))
                active_ids = set(cur_tracks['players'].keys())
                stale_ids = active_jersey_ids - active_ids
                for stale_id in stale_ids:
                    jersey_ocr.reset_history(stale_id)
                active_jersey_ids = active_ids

                tracker.sync_display_assignments(jersey_display_map, new_player_id_map)

                active_source_tracks = {
                    info.get("source_track_id")
                    for info in cur_tracks['players'].values()
                    if info.get("source_track_id") is not None
                }
                for track_id in list(track_jersey_cache.keys()):
                    meta = track_jersey_cache.get(track_id) or {}
                    last_frame = meta.get("frame", -1)
                    if (
                        track_id not in active_source_tracks
                        and frame_idx - last_frame > JERSEY_CACHE_TTL_FRAMES
                    ):
                        track_jersey_cache.pop(track_id, None)
                for jersey_key in list(jersey_track_cache.keys()):
                    jersey_meta = jersey_track_cache.get(jersey_key) or {}
                    if frame_idx - jersey_meta.get("frame", -1) > JERSEY_CACHE_TTL_FRAMES:
                        jersey_track_cache.pop(jersey_key, None)

                for display_id, info in cur_tracks['players'].items():
                    jersey = info.get('jersey_number')
                    bbox = info.get('bbox')
                    if jersey and bbox:
                        x1, y1, x2, y2 = map(float, bbox)
                        center_x = 0.5 * (x1 + x2)
                        center_y = 0.5 * (y1 + y2)
                        history = jersey_position_history.get(jersey)
                        if history is None:
                            history = deque(maxlen=20)
                            jersey_position_history[jersey] = history
                        history.append((frame_idx, center_x, center_y))
            else:
                if active_jersey_ids:
                    for stale_id in active_jersey_ids:
                        jersey_ocr.reset_history(stale_id)
                    active_jersey_ids = set()
                cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))
        else:
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))
            active_jersey_ids = set(cur_tracks['players'].keys())

        # Fit team 1 lần khi có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_model(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        # Gán team & màu vào tracks 
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            info['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

        vt.update_homography_from_frame(frame)
        
        id2color, id2label, pid2team = {}, {}, {}

        # players
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            color_bgr = tuple(int(c) for c in team_assigner.team_colors.get(team, (0, 0, 255)))
            info['team_color'] = color_bgr
            id2color[pid] = color_bgr
            jersey_text = info.get('jersey_number')
            id2label[pid] = jersey_text if jersey_text else pid
            pid2team[pid] = team
        
        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        drawn = draw_jersey_labels(drawn, cur_tracks['players'])
        
        # camera movement
        if cm_est is None:
            cm_est = CameraMovementEstimator(frame)
            dx, dy = 0.0, 0.0
        else:
            dx, dy = cm_est.update(frame)

        drawn = cm_est.draw_overlay(drawn, dx, dy)

        # minimap
        id2color, id2label, pid2team = {}, {}, {}

        # players
        TEAM_COLORS = team_assigner.team_colors
        for pid, info in cur_tracks['players'].items():
            team = info.get('team', None)
            color_bgr = tuple(int(c) for c in TEAM_COLORS.get(team, (0, 0, 255)))
            id2color[pid] = color_bgr
            id2label[pid] = pid
            pid2team[pid] = team

        # refs
        REF_COLOR = (0, 255, 255)
        if 'refs' in cur_tracks and cur_tracks['refs']:
            for rid, info in cur_tracks['refs'].items():
                id2color[rid] = REF_COLOR
                id2label[rid] = f"R{rid}"

        # update history, render & blend
        pid_feet_players = vt.compute_feet(w, h, cur_tracks['players'])
        pid_feet_refs    = vt.compute_feet(w, h, cur_tracks.get('refs', {}))
        pid_feet_all     = {**pid_feet_players, **pid_feet_refs}

        # only ingest when homography is valid
        if getattr(cam_calib, "has_valid_h", False):
            vt.update_history(pid_feet_all, frame_idx)

        # bóng
        ball_xy = None
        if 'ball' in cur_tracks and cur_tracks['ball']:
            ball_xy = vt.compute_ball(w, h, cur_tracks['ball'])
        vt.update_ball(ball_xy)

        field_img = vt.render_minimap(id2color=id2color, id2label=id2label)
        drawn = vt.blend_to_frame(drawn, field_img)
        drawn = draw_id_mapping_overlay(drawn, jersey_display_map)

        # Action Spotting (tạm tắt)
        if ball_action is not None:
            drawn = ball_action.visualize_frame(drawn, frame_idx)

        write_frame(writer, drawn)
        frame_idx += 1

        if frame_idx >= fps * 300:
            break

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Wrote: {out_path}")


if __name__ == '__main__':
    main()
