import cv2
from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer

def main():
    in_path  = 'input_videos/cut20s.mp4'
    out_path = 'output_videos/cut20s_test.avi'

    cap, fps, w, h = open_video(in_path)
    tracker = Tracker('models/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner()

    writer = create_writer(out_path, fps, w, h)
    team_fit_done = False

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        # Fit team 1 lần khi có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_color(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        # Gán team & màu vào tracks (để draw_* lấy ra)
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            info['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

        # Draw
        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        write_frame(writer, drawn)

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Wrote: {out_path}")

if __name__ == '__main__':
    main()
