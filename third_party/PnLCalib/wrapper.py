# third_party/PnLCalib/wrapper.py
from __future__ import annotations

import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from .model.cls_hrnet import get_cls_net
from .model.cls_hrnet_l import get_cls_net as get_cls_net_l
from .utils.utils_calib import FramebyFrameCalib
from .utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)


class PnLCalibWrapper:

    def __init__(
        self,
        kp_weight: str,
        line_weight: str,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
        device: str = "cuda",
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> None:

        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.kp_threshold = float(kp_threshold)
        self.line_threshold = float(line_threshold)
        self.pnl_refine = bool(pnl_refine)

        self.H_img2pitch: Optional[np.ndarray] = None
        self.has_valid_h: bool = False

        root = Path(__file__).resolve().parent
        cfg_path = root / "config" / "hrnetv2_w48.yaml"
        cfg_l_path = root / "config" / "hrnetv2_w48_l.yaml"

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg_l = yaml.safe_load(cfg_l_path.read_text(encoding="utf-8"))

        self.model_kp = get_cls_net(cfg)
        state_kp = torch.load(kp_weight, map_location=self.device)
        self.model_kp.load_state_dict(state_kp)
        self.model_kp.to(self.device)
        self.model_kp.eval()

        self.model_line = get_cls_net_l(cfg_l)
        state_ln = torch.load(line_weight, map_location=self.device)
        self.model_line.load_state_dict(state_ln)
        self.model_line.to(self.device)
        self.model_line.eval()

        self.resize = T.Resize((540, 960))

        iw = img_width if img_width is not None else 960
        ih = img_height if img_height is not None else 540
        self.cam = FramebyFrameCalib(iwidth=iw, iheight=ih, denormalize=True)

    def _run_network(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        tensor = F.to_tensor(pil_img).float().unsqueeze(0)
        _, _, h0, w0 = tensor.size()

        if tensor.size(-1) != 960:
            tensor = self.resize(tensor)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            heatmaps_kp = self.model_kp(tensor)
            heatmaps_ln = self.model_line(tensor)

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_ln[:, :-1])

        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        line_dict = coords_to_dict(line_coords, threshold=self.line_threshold)

        kp_dict, line_dict = complete_keypoints(
            kp_dict[0], line_dict[0], w=960, h=540, normalize=True
        )

        return kp_dict, line_dict

    def compute_H(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:

        kp_dict, line_dict = self._run_network(frame_bgr)

        self.cam.update(kp_dict, line_dict)

        try:
            res = self.cam.heuristic_voting_ground(refine_lines=self.pnl_refine)
        except Exception as e:
            print("[PnLCalib] ERROR:", e)
            return None

        if res is None:
            self.has_valid_h = False
            self.H_img2pitch = None
            return None

        H = res.get("homography", None)
        if H is None:
            self.has_valid_h = False
            self.H_img2pitch = None
            return None

        self.H_img2pitch = H
        self.has_valid_h = True
        return H
