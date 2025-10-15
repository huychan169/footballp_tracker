import cv2
from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
# from ball_tracker import Ball_tracker

def main():
    in_path  = 'input_videos/match1_clip_104_cfr.mp4'
    out_path = 'output_videos/match1_clip_104_cfr_1015n.avi'

    # Stream
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {in_path}")

    tracker = Tracker('models/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner()

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    team_fit_done = False
    # ball_buffer = []

    # đọc -> detect/track -> team + interpolate -> draw
    while True:
        ret, frame = cap.read()
        if not ret:  # hết video
            break

        # VideoWriter ở frame đầu (sau khi đã có frame thật để lấy w,h)
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open writer for {out_path}")

        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}  

        # # interpolation
        # ball_buffer.append(cur_tracks['ball'])
        # if len(ball_buffer) >= 30:
        #     ball_buffer = tracker.interpolate_ball_positions(ball_buffer)  
        #     cur_tracks['ball'] = ball_buffer[-1]

        # Fit màu đội 1 lần khi đã có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_color(frame, cur_tracks['players'])
            # có guard, check kmeans fit
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        # Gán team 
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            color = team_assigner.team_colors.get(team, (0, 0, 255)) 
            frame = tracker.draw_ellipse(frame, info['bbox'], color, pid)

        # Draw
        for _, rf in cur_tracks['refs'].items():
            frame = tracker.draw_ellipse(frame, rf['bbox'], (0, 255, 255)) 
        for _, bl in cur_tracks['ball'].items():
            frame = tracker.draw_bbox_with_id(frame, bl['bbox'], (0, 255, 0))  

        writer.write(frame)
        

    cap.release()
    if writer is not None:
        writer.release()
    print(f"[DONE] Wrote: {out_path}")


if __name__ == '__main__':
    main()
