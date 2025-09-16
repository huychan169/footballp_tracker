from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video 
    video_frames = read_video('input_videos\match1_clip_104.mp4')

    # Initialize Tracker
    tracker = Tracker('models\last_v5lu_ep30_cl.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/match1_clip_104_v5l_v2.avi')

if __name__ == '__main__':
    main()
