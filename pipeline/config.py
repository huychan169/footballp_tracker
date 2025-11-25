from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = Path("input_videos/video_cut_3.mp4")
    output_path: Path = Path("output_videos/video_cut_3_new.avi")
    profile_output: Path = Path("output_videos/pipeline_profile.txt")
    tracker_model_path: Path = Path("models/best_ylv8_ep50.pt")
    reid_weights: Path = Path("reid/models/model.pth.tar-30")
    parseq_checkpoint: Path = Path("outputs/parseq/03_11_2025/checkpoints/test.ckpt")
    pose_model_path: Path = Path("models/yolov8m-pose.pt")
    ocr_frame_stride: int = 2
    jersey_cache_ttl_frames: int = 150
    jersey_cache_min_confidence: float = 0.58
    homography_update_every: int = 3
    max_runtime_seconds: int = 300
    ball_trail_suffix: str = "_ball_trail.mp4"
    use_boost: bool = True
    use_reid: bool = True
    reid_backend: str = "torchreid"


def build_default_config() -> PipelineConfig:
    return PipelineConfig()
