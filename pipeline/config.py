from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = Path("data/inputs/video_cut_3.mp4")
    output_path: Path = Path("data/outputs/video_cut_3_new.avi")
    profile_output: Path = Path("data/outputs/pipeline_profile.txt")
    # Tracker config
    tracker_model_path: Path = Path("data/models/best_ylv8_ep50.pt")
    reid_weights: Path = Path("data/models/reid/model/model.pth.tar-30")
    use_boost: bool = True
    use_reid: bool = False
    reid_backend: str = "torchreid"
    # OCR and Jersey config
    parseq_checkpoint: Path = Path("data/models/parseq/03_11_2025/checkpoints/test.ckpt")
    pose_model_path: Path = Path("data/models/yolov8m-pose.pt")
    ocr_frame_stride: int = 2
    jersey_cache_ttl_frames: int = 150
    jersey_cache_min_confidence: float = 0.58
    # Homography config
    homography_update_every: int = 3
    line_path: Path = None # Path("data/models/line_extreme_heatpoints2/model-013-0.826654.pth")
    hrnet_path: Path = Path("data/models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth")

    max_runtime_seconds: int = 300
    ball_trail_suffix: str = "_ball_trail.mp4"


def build_default_config() -> PipelineConfig:
    return PipelineConfig()
