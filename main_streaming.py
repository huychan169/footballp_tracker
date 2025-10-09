import cv2
from trackers import Tracker
from utils_stream import frame_reader, video_writer
from team_assigner.team_assigner import TeamAssigner  

def main():
    in_path = 'input_videos/match1_clip_104_cfr.mp4'
    out_path = 'output_videos/match1_clip_104_cfr_108n.avi'

    # Lấy kích thước và fps từ metadata
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = video_writer(out_path, fps=fps, w=w, h=h)

    tracker = Tracker('models/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner()

    last_ball_bbox = None
    team_fit_done = False

    # Streaming loop
    for frame_idx, frame, cur_tracks in tracker.stream_tracks(
        frame_iter=frame_reader(in_path), batch_size=16
    ):
        if not team_fit_done and len(cur_tracks['players']) > 0:
            team_assigner.assign_team_color(frame, cur_tracks['players'])
            team_fit_done = True

        # Gán team
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            if team is not None and team in team_assigner.team_colors:
                info['team'] = team
                info['team_color'] = team_assigner.team_colors[team]
            else:
                info['team_color'] = (0, 0, 255)

        # Nội suy bóng đơn giản (last-seen)
        if 1 in cur_tracks['ball']:
            last_ball_bbox = cur_tracks['ball'][1]['bbox']
        elif last_ball_bbox is not None:
            cur_tracks['ball'][1] = {'bbox': last_ball_bbox}

        # Draw
        for pid, pl in cur_tracks['players'].items():
            frame = tracker.draw_ellipse(frame, pl['bbox'], pl['team_color'], pid)
        for _, rf in cur_tracks['refs'].items():
            frame = tracker.draw_ellipse(frame, rf['bbox'], (0, 255, 255))
        for _, bl in cur_tracks['ball'].items():
            frame = tracker.draw_bbox_with_id(frame, bl['bbox'], (0, 255, 0))

        writer.write(frame)

    writer.release()
    print(f"Done: {out_path}")


if __name__ == '__main__':
    main()
