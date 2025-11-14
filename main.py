from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from view_transformer import ViewTransformer2D
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from camera_movement_estimator import CameraMovementEstimator

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend

from BallAction.Test_Visual import BallActionSpot

def main():
    in_path  = 'video/input_videos/match1_cut45s.mp4'
    out_path = 'video/output_videos/match1_cut45s_1114v.avi'

    cap, fps, w, h = open_video(in_path)
    tracker = Tracker('data/models/detect/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner(flip_guard_frames=8)

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
        ema_alpha=0.4,
        max_step=10,
        margin=4
    )

    # offset tích lũy chuyển động camera 
    cam_off_x = 0.0
    cam_off_y = 0.0

    ball_action = BallActionSpot()

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        # Fit team 1 lần khi có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_model(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        vt.update_homography_from_frame(frame)

        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        
        # camera movement
        if cm_est is None:
            cm_est = CameraMovementEstimator(frame)
            dx, dy = 0.0, 0.0
        else:
            dx, dy = cm_est.update(frame)

        # tích luỹ offset camera
        cam_off_x += dx
        cam_off_y += dy

        drawn = cm_est.draw_overlay(drawn, dx, dy)

        # minimap
        id2color, id2label, pid2team = {}, {}, {}

        # players
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            color_bgr = tuple(int(c) for c in team_assigner.team_colors.get(team, (0, 0, 255)))
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

        # scale sang IMG_W, IMG_H
        scale_x = cam_calib.IMG_W / w
        scale_y = cam_calib.IMG_H / h

        for pid, (x, y) in list(pid_feet_all.items()):
            x_stab = x - cam_off_x * scale_x
            y_stab = y - cam_off_y * scale_y
            pid_feet_all[pid] = (x_stab, y_stab)

        vt.update_history(pid_feet_all)

        # bóng
        ball_xy = None
        if 'ball' in cur_tracks and cur_tracks['ball']:
            ball_xy = vt.compute_ball(w, h, cur_tracks['ball'])
        vt.update_ball(ball_xy)

        field_img = vt.render_minimap(id2color=id2color, id2label=id2label)
        drawn = vt.blend_to_frame(drawn, field_img)

        # Action Spotting
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
