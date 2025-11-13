from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from view_transformer import ViewTransformer2D
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from camera_movement_estimator import CameraMovementEstimator

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend

# from BallAction.Test_Visual import BallActionSpot

def main():
    in_path  = 'video/input_videos/match1_clip_105_cfr.mp4'
    out_path = 'video/output_videos/match1_clip_105_1113.avi'

    cap, fps, w, h = open_video(in_path)
    tracker = Tracker('data/models/detect/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner(flip_guard_frames=8)

    writer = create_writer(out_path, fps, w, h)
    team_fit_done = False
    cm_est = None
    frame_idx = 0
    cam_calib = CamCalib(MODEL_PATH, LINES_FILE)
    vt = ViewTransformer2D(cam_calib, scale=0.5, alpha=0.4,
                        dot_radius=8, ema_alpha=0.3, max_step=10, margin=6)

    # ball_action = BallActionSpot()

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        # Fit team 1 lần khi có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_model(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None


        # Gán team & màu vào tracks
        feets = []
        colors = []
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            info['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / w, y1 / h, x2 / w, y2 / h
            feets.append(cam_calib.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n)))
            colors.append(info['team_color'])

        vt.update_homography_from_frame(frame) #add

        # Draw
        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        
        # camera movement
        if cm_est is None:                            
            cm_est = CameraMovementEstimator(frame)  
            dx, dy = 0.0, 0.0                        
        else:
            dx, dy = cm_est.update(frame)
        
        drawn = cm_est.draw_overlay(drawn, dx, dy)  

        id2color, id2label, pid2team = {}, {}, {}
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            color_bgr = tuple(int(c) for c in team_assigner.team_colors.get(team, (0, 0, 255)))
            id2color[pid] = color_bgr
            id2label[pid] = pid
            pid2team[pid] = team

        # chiếu 2D, cập nhật lịch sử, render & blend
        pid_feet = vt.compute_feet(w, h, cur_tracks['players'])
        vt.update_history(pid_feet)
        field_img = vt.render_minimap(id2color=id2color, id2label=id2label)
        drawn = vt.blend_to_frame(drawn, field_img)

        # Action Spotting
        # drawn = ball_action.visualize_frame(drawn, frame_idx)

        write_frame(writer, drawn)
        frame_idx += 1
        
        if frame_idx >= fps * 300: 
            break

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Wrote: {out_path}")


if __name__ == '__main__':
    main()
