from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from tqdm import tqdm

from Homography.field_markings.run import CamCalib
from DetectTrack.camera_movement_estimator import CameraMovementEstimator
from DetectTrack.team_assigner.team_assigner import TeamAssigner
from DetectTrack.trackers import Tracker
from utils.video_utils import close_video, close_writer, create_writer, iter_frames, open_video, write_frame
from Homography.view_transformer import ViewTransformer2D
from BallAction import BallActionModel

from utils.export_ball_trail_video import export_ball_trail_video
from pipeline.ball import select_ball_detection
from pipeline.config import PipelineConfig
from pipeline.jersey import JerseyCoordinator
from pipeline.overlays import draw_id_mapping_overlay, draw_jersey_labels


def run_pipeline(config: PipelineConfig):
    cap, fps, w, h, total_frames = open_video(str(config.input_path))
    tracker = Tracker(
        str(config.tracker_model_path),
        use_boost=config.use_boost,
        use_reid=config.use_reid,
        reid_backend=config.reid_backend,
        torchreid_weights=str(config.reid_weights),
    )
    team_assigner = TeamAssigner(flip_guard_frames=8)
    jersey = JerseyCoordinator(tracker, config)

    jersey_display_map = jersey.display_map
    timings = defaultdict(float)
    counts = defaultdict(int)

    writer = create_writer(str(config.output_path), fps, w, h)
    team_fit_done = False
    cm_est = None
    frame_idx = 0
    cam_calib = CamCalib(config.hrnet_path, config.line_path)
    tail_frames = int(max(25, min(50, round(fps * 2)))) if fps else 50
    vt = ViewTransformer2D(
        cam_calib,
        scale=0.5,
        alpha=0.4,
        dot_radius=8,
        ema_alpha=0.4,
        max_step=10,
        margin=4,
        ball_tail=tail_frames,
        ball_tail_long=200,
        ball_ttl=max(200, tail_frames),
    )

    ball_action = BallActionModel(config.ballaction_path, config.ballaction_conf, fps)
    ball_trail_records = []
    prev_ball = deque(maxlen=int(fps//2))

    for frame in tqdm(iter_frames(cap), total=total_frames):
        loop_start = perf_counter()
        t0 = perf_counter()
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        timings["tracking"] += perf_counter() - t0
        counts["tracking"] += 1
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        cur_tracks = jersey.process(frame, cur_tracks, frame_idx, timings, counts)

        if not team_fit_done and cur_tracks['players']:
            t0 = perf_counter()
            team_assigner.assign_team_model(frame, cur_tracks['players'])
            timings["team_fit"] += perf_counter() - t0
            counts["team_fit"] += 1
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        if config.enable_homography and (frame_idx % config.homography_update_every == 0 or cam_calib.H is None):
            t0 = perf_counter()
            vt.update_homography_from_frame(frame)
            timings["homography"] += perf_counter() - t0
            counts["homography"] += 1

        id2color, id2label, pid2team, id2jersey = {}, {}, {}, {}

        t0 = perf_counter()
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            color_bgr = tuple(int(c) for c in team_assigner.team_colors.get(team, (0, 0, 255)))
            info['team_color'] = color_bgr
            id2color[pid] = color_bgr
            jersey_text = info.get('jersey_number')
            id2label[pid] = jersey_text if jersey_text else pid
            id2jersey[pid] = jersey_text if jersey_text else None
            pid2team[pid] = team

        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        drawn = draw_jersey_labels(drawn, cur_tracks['players'])
        timings["team_assign"] += perf_counter() - t0
        counts["team_assign"] += 1

        t0 = perf_counter()
        if cm_est is None:
            cm_est = CameraMovementEstimator(frame)
            dx, dy = 0.0, 0.0
        else:
            dx, dy = cm_est.update(frame)

        drawn = cm_est.draw_overlay(drawn, dx, dy)
        timings["camera_motion"] += perf_counter() - t0
        counts["camera_motion"] += 1

        REF_COLOR = (0, 255, 255)
        if 'refs' in cur_tracks and cur_tracks['refs']:
            for rid, info in cur_tracks['refs'].items():
                id2color[rid] = REF_COLOR
                id2label[rid] = f"R{rid}"

        t0 = perf_counter()
        pid_feet_players = vt.compute_feet(w, h, cur_tracks['players'])
        pid_feet_refs = vt.compute_feet(w, h, cur_tracks.get('refs', {}))
        pid_feet_all = {**pid_feet_players, **pid_feet_refs}

        if getattr(cam_calib, "has_valid_h", False):
            vt.update_history(pid_feet_all, frame_idx)
        timings["feet_projection"] += perf_counter() - t0
        counts["feet_projection"] += 1

        t0 = perf_counter()
        ball_xy = None
        ball_keep_last = False
        ball_air = not getattr(cam_calib, "has_valid_h", False)
        ball_xy_uncalibrated = None

        if 'ball' in cur_tracks and cur_tracks['ball']:
            best_ball = select_ball_detection(cur_tracks['ball'], w, h)
            if best_ball:
                area = best_ball.get("area")
                ball_xy_uncalibrated = vt.compute_ball_uncalibrated(w, h,
                                                                    best_ball,
                                                                    use_center=True,
                                                                    enable_homography=False)
                if area is not None and area > 0.006 * w * h:
                    ball_air = True
                if not ball_air:
                    ball_xy = vt.compute_ball(w, h, best_ball, use_center=True)
                else:
                    ball_keep_last = True

        if ball_air and ball_xy is None:
            ball_keep_last = True

        vt.update_ball(ball_xy, frame_idx, is_airball=ball_air, keep_last=ball_keep_last)
        ball_trail_records.append({
            "xy": ball_xy,
            "air": ball_air,
            "keep_last": ball_keep_last,
        })

        timings["ball"] += perf_counter() - t0
        counts["ball"] += 1
        t0 = perf_counter()
        field_img = vt.render_minimap(id2color=id2color, id2label=id2label)
        drawn = vt.blend_to_frame(drawn, field_img)
        drawn = draw_id_mapping_overlay(drawn, jersey_display_map)
        timings["minimap_render"] += perf_counter() - t0
        counts["minimap_render"] += 1
        
        pid_feet_players_uncalibrated = vt.compute_feet_uncalibrated(w, h, cur_tracks['players'], enable_homography=False)
        possession_team = possession_player = None

        prev_ball.append(ball_xy_uncalibrated)
        prev_ball_filtered = [b for b in prev_ball if b is not None]
        ball_xy_choice = prev_ball_filtered[-1] if len(prev_ball_filtered) > 0 else None

        if ball_xy_choice is not None:
            min_dist = float('inf')

            for feet_id, feet_xy in pid_feet_players_uncalibrated.items():
                dist = ((feet_xy[0] - ball_xy_choice[0]) ** 2\
                        + (feet_xy[1] - ball_xy_choice[1]) ** 2) ** 0.5

                if dist < min_dist and dist < 60:  # meters
                    min_dist = dist
                    possession_player = id2jersey.get(feet_id, None)
                    possession_team = pid2team.get(feet_id, None)

                    if possession_team is not None:
                        possession_team = "team1" if possession_team == 1 else "team2"

        drawn = ball_action.visualize_frame(drawn, frame_idx, possession_team, possession_player)

        t0 = perf_counter()
        write_frame(writer, drawn)
        timings["write_frame"] += perf_counter() - t0
        counts["write_frame"] += 1
        frame_idx += 1
        timings["frame_total"] += perf_counter() - loop_start
        counts["frame_total"] += 1

        if fps and frame_idx >= fps * config.max_runtime_seconds:
            break

    _write_profile(config.profile_output, fps, frame_idx, timings, counts)
    _export_ball_trail(config, fps, ball_trail_records, vt)

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Wrote: {config.output_path}")


def _write_profile(profile_output: Path, fps: float, frame_idx: int, timings, counts):
    try:
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        frame_count = counts.get("frame_total", frame_idx)
        lines = [
            f"frames: {frame_count}",
            f"video_fps: {fps}",
        ]
        for key, total in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
            calls = max(1, counts.get(key, 1))
            avg_call = total / calls * 1000.0
            avg_per_frame = total / max(1, frame_count) * 1000.0
            lines.append(
                f"{key}: total={total:.2f}s avg_frame={avg_per_frame:.2f}ms avg_call={avg_call:.2f}ms calls={calls}"
            )
        profile_output.write_text("\n".join(lines), encoding="utf-8")
        print(f"[PROFILE] Wrote {profile_output}")
    except Exception as exc:
        print(f"[PROFILE] Failed to write profile: {exc}")


def _export_ball_trail(config: PipelineConfig, fps: float, ball_trail_records, vt: ViewTransformer2D):
    try:
        output_path = config.output_path
        trail_path = output_path.with_name(output_path.stem + config.ball_trail_suffix)
        export_ball_trail_video(
            ball_records=ball_trail_records,
            vt=vt,
            output_path=trail_path,
            fps=fps,
            trail_frames=200,
        )
    except Exception as exc:
        print(f"[BallTrail] Xuất video ball trail lỗi: {exc}")
