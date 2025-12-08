from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = Path("data/inputs/video_cut_3.mp4")
    output_path: Path = Path("data/outputs/video_cut_3.avi")
    profile_output: Path = Path("data/outputs/pipeline_profile.txt")

    # Tracker config
    tracker_model_path: Path = Path("data/models/best_ylv8_ep50.pt")
    reid_weights: Path = Path("data/models/reid/model/model.pth.tar-30")
    use_boost: bool = True
    use_reid: bool = False
    reid_backend: str = "torchreid"

    # OCR and Jersey config
    parseq_checkpoint: Path = Path("data/models/parseq/04_12_2025/checkpoints/test.ckpt")
    pose_model_path: Path = Path("data/models/yolo11n-pose.pt")
    enable_pose_crop: bool = True
    ocr_frame_stride: int = 1
    ocr_enable_crop_debug: bool = False
    ocr_crop_dir: Path = Path("data/outputs/croped")
    ocr_crop_limit: int = 0  # 0 = save all crops when debug is enabled
    ocr_cache_refresh_stride: int = 25  # force re-OCR cached players every N frames to correct mistakes
    ocr_low_conf_threshold: float = 0.6
    ocr_low_consensus_threshold: float = 0.55
    ocr_history_window: int = 16
    ocr_hard_age_cutoff: int = 40
    ocr_change_persist_frames: int = 3  # require N hits before switching number
    ocr_overwrite_margin: float = 0.1   # allow immediate switch if conf improves by this margin
    jersey_cache_ttl_frames: int = 150
    jersey_cache_min_confidence: float = 0.58

    # Homography config
    enable_homography: bool = False
    homography_update_every: int = 3
    line_path: Path = None # Path("data/models/line_extreme_heatpoints2/model-013-0.826654.pth")
    hrnet_path: Path = Path("data/models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth")

    # Ball Action config
    ballaction_path: Path = Path("data/models/ball_action/results_spotting.json")
    ballaction_teamlist: Path = None
    ballaction_conf: float = 0.2

    # Misc config
    max_runtime_seconds: int = 300
    ball_trail_suffix: str = "_ball_trail.mp4"


def build_default_config() -> PipelineConfig:
    return PipelineConfig()
