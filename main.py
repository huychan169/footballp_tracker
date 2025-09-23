from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video 
    video_frames = read_video('input_videos/match1_clip_104.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best_ylv8_ep50.pt', use_boost=True)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/match1_clip_104_ylv8_bs_el.avi')

if __name__ == '__main__':
    main()
