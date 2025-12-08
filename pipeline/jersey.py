import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Deque, Dict, Optional, Set, Tuple

import cv2
import torch
from PIL import Image

from JerseyNumber.ocr.jersey_recognizer import (
    OCRReading,
    JerseyRecogniser,
    _compute_confidence,
)

from pipeline.config import PipelineConfig


class JerseyCoordinator:
    def __init__(self, tracker, config: PipelineConfig):
        self.config = config
        self.tracker = tracker
        self.jersey_display_map: Dict[str, int] = {}
        self.active_jersey_ids: Set[int] = set()
        self.jersey_position_history: Dict[str, Deque[Tuple[int, float, float]]] = {}
        self.track_jersey_cache: Dict[int, Dict[str, float]] = {}
        self.jersey_track_cache: Dict[str, Dict[str, int]] = {}
        self.jersey_ocr = self._build_recogniser()
        self.last_ocr_frame: Dict[int, int] = {}
        self.last_consensus: Dict[int, float] = {}
        self.track_to_number: Dict[int, Optional[str]] = {}
        self.track_number_conf: Dict[int, float] = {}
        self.number_history: Dict[int, list] = defaultdict(list)
        self.occurrences_per_number: Dict[str, Set[int]] = defaultdict(set)
        self.pending_number: Dict[int, Dict[str, float]] = {}

    def _build_recogniser(self) -> Optional[JerseyRecogniser]:
        tracker_device = getattr(self.tracker, "device", "cuda:0")
        jersey_device = "cuda" if str(tracker_device).startswith("cuda") else "cpu"
        crop_dir = self.config.ocr_crop_dir if self.config.ocr_enable_crop_debug else None
        crop_limit = self.config.ocr_crop_limit if self.config.ocr_enable_crop_debug else 0
        return JerseyRecogniser(
            parseq_root=Path.cwd(),
            checkpoint_path=self.config.parseq_checkpoint,
            pose_model_path=self.config.pose_model_path,
            device=jersey_device,
            enable_pose_crop=self.config.enable_pose_crop,
            crop_debug_dir=crop_dir,
            crop_debug_limit=crop_limit,
            history_window=self.config.ocr_history_window,
            confidence_threshold=0.6,
            vote_min_confidence=0.55,
            vote_min_support=3,
            vote_high_threshold=0.6,
            vote_count_min=3,
            vote_count_margin=2,
            hard_age_cutoff=self.config.ocr_hard_age_cutoff,
        )

    def process(self, frame, cur_tracks, frame_idx: int, timings, counts):
        if self.jersey_ocr is None:
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))
            self.active_jersey_ids = set(cur_tracks['players'].keys())
            return cur_tracks

        t0 = perf_counter()
        if cur_tracks['players']:
            player_entries = list(cur_tracks['players'].items())
            attempt_entries = []
            player_meta = []

            if self.config.ocr_enable_crop_debug and self.jersey_ocr.crop_debug_dir is not None:
                # Re-enable debug saving if it was disabled earlier.
                self.jersey_ocr.enable_crop_debug(self.jersey_ocr.crop_debug_dir, limit=self.config.ocr_crop_limit, reset=False)

            for display_id, info in player_entries:
                bbox = info.get('bbox')
                source_tid = info.get('source_track_id')
                center = None
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                attempt = self._should_attempt_ocr(display_id, source_tid, info, bbox, frame_idx)
                if attempt and bbox:
                    attempt_entries.append({
                        "display_id": display_id,
                        "bbox": tuple(map(int, bbox)),
                        "info": info,
                    })
                player_meta.append((display_id, info, bbox, source_tid, center, attempt))

            pose_crops = {}
            if attempt_entries and self.config.enable_pose_crop:
                pose_entries = [(entry["display_id"], entry["bbox"]) for entry in attempt_entries]
                pose_crops = self.jersey_ocr._pose_crops_batch(frame, pose_entries)

            batch_readings = self._batch_read_numbers(frame, attempt_entries, pose_crops, frame_idx) if attempt_entries else {}

            for display_id, info, bbox, source_tid, center, attempt in player_meta:
                reading = batch_readings.get(display_id) if (attempt and bbox) else None
                if reading is not None:
                    key = source_tid if source_tid is not None else display_id
                    self.last_ocr_frame[key] = frame_idx
                    self._update_number_with_hysteresis(key, reading, frame_idx)
                jersey_assigned = self.track_to_number.get(source_tid if source_tid is not None else display_id)
                info.pop('jersey_number', None)
                info.pop('jersey_confidence', None)
                if jersey_assigned:
                    info['jersey_number'] = jersey_assigned
                    info['jersey_confidence'] = (
                        reading.confidence
                        if reading and jersey_assigned == (reading.text if reading else None)
                        else self.track_number_conf.get(source_tid if source_tid is not None else display_id)
                    )
                    self.jersey_display_map[jersey_assigned] = display_id

            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))
            active_ids = set(cur_tracks['players'].keys())
            stale_ids = self.active_jersey_ids - active_ids
            for stale_id in stale_ids:
                self.jersey_ocr.reset_history(stale_id)
            self.active_jersey_ids = active_ids
        else:
            if self.active_jersey_ids:
                for stale_id in self.active_jersey_ids:
                    self.jersey_ocr.reset_history(stale_id)
                self.active_jersey_ids = set()
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))

        timings["jersey_ocr"] += perf_counter() - t0
        counts["jersey_ocr"] += 1
        return cur_tracks

    def _should_attempt_ocr(self, display_id: int, source_tid: Optional[int], info, bbox, frame_idx: int) -> bool:
        if bbox is None:
            return False
        stride_hit = (self.config.ocr_frame_stride <= 1) or (frame_idx % self.config.ocr_frame_stride == 0)
        missing_jersey = ('jersey_number' not in info)
        debug_capture = self.config.ocr_enable_crop_debug
        last_key = source_tid if source_tid is not None else display_id
        last_ocr = self.last_ocr_frame.get(last_key, -1)
        refresh_gap = (last_ocr < 0) or (frame_idx - last_ocr >= self.config.ocr_cache_refresh_stride)
        consensus = self.last_consensus.get(last_key, 1.0)
        low_vote = consensus < self.config.ocr_low_consensus_threshold
        return (
            missing_jersey
            or debug_capture
            or stride_hit
            or refresh_gap
            or low_vote
        )

    def _batch_read_numbers(self, frame, attempt_entries, pose_crops, frame_idx: int) -> Dict[int, OCRReading]:
        if not attempt_entries:
            return {}
        tensors = []
        entry_refs = []
        for entry in attempt_entries:
            display_id = entry["display_id"]
            bbox = entry["bbox"]
            patch = pose_crops.get(display_id)
            crop_source = "pose" if patch is not None else "bbox"
            if patch is None:
                patch = self._enlarge_and_crop(frame, bbox, scale=1.12)
            if patch is None:
                continue
            self.jersey_ocr._save_debug_patch(patch, frame_idx, crop_source, bbox)
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_patch)
            tensor = self.jersey_ocr.transform(image)
            tensors.append(tensor)
            entry_refs.append(display_id)

        if not tensors:
            return {}

        batch = torch.stack(tensors).to(self.jersey_ocr.device)
        with torch.inference_mode():
            logits = self.jersey_ocr.model(batch)
            probs = logits.softmax(-1)
            labels, confidences = self.jersey_ocr.model.tokenizer.decode(probs)

        readings: Dict[int, OCRReading] = {}
        for idx, display_id in enumerate(entry_refs):
            label = labels[idx] if idx < len(labels) else ""
            conf_tensor = confidences[idx] if idx < len(confidences) else None
            reading = self._parse_reading(label, conf_tensor)
            if reading is not None:
                readings[display_id] = reading
        return readings

    def _parse_reading(self, label: str, conf_tensor) -> Optional[OCRReading]:
        if not label or label.strip() == "~":
            return None
        confidence = _compute_confidence(conf_tensor) if conf_tensor is not None else 0.0
        if confidence < self.jersey_ocr.confidence_threshold:
            return None
        clean = "".join(ch for ch in label if ch.isdigit())
        if not clean or not clean.isdigit():
            return None
        value = int(clean)
        if value <= 0 or value > 99:
            return None
        return OCRReading(text=f"{value:02d}", confidence=confidence)

    def _enlarge_and_crop(self, frame, bbox, scale=1.12):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        new_x1 = int(max(cx - 0.5 * w, 0))
        new_y1 = int(max(cy - 0.5 * h, 0))
        new_x2 = int(min(cx + 0.5 * w, frame.shape[1] - 1))
        new_y2 = int(min(cy + 0.5 * h, frame.shape[0] - 1))
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            return None
        return frame[new_y1:new_y2, new_x1:new_x2]

    def _update_number_with_hysteresis(self, key: int, reading: OCRReading, frame_idx: int):
        current = self.track_to_number.get(key)
        current_conf = self.track_number_conf.get(key, 0.0)
        if current is None:
            self.track_to_number[key] = reading.text
            self.track_number_conf[key] = reading.confidence
            self.number_history[key].append((frame_idx, reading.text))
            self.occurrences_per_number[reading.text].add(key)
            self.last_consensus[key] = reading.confidence
            self.pending_number.pop(key, None)
            return

        if reading.text == current:
            # Reinforce confidence and clear pending.
            self.track_number_conf[key] = max(current_conf, reading.confidence)
            self.last_consensus[key] = reading.confidence
            self.pending_number.pop(key, None)
            return

        pending = self.pending_number.get(key)
        if pending and pending.get("candidate") == reading.text:
            pending["count"] = pending.get("count", 1) + 1
            pending["best_conf"] = max(pending.get("best_conf", 0.0), reading.confidence)
        else:
            pending = {"candidate": reading.text, "count": 1, "best_conf": reading.confidence}
        self.pending_number[key] = pending

        if (
            pending["count"] >= self.config.ocr_change_persist_frames
            or reading.confidence >= current_conf + self.config.ocr_overwrite_margin
        ):
            self.track_to_number[key] = pending["candidate"]
            self.track_number_conf[key] = pending["best_conf"]
            self.number_history[key].append((frame_idx, pending["candidate"]))
            self.occurrences_per_number[pending["candidate"]].add(key)
            self.last_consensus[key] = pending["best_conf"]
            self.pending_number.pop(key, None)

    def reset(self):
        self.jersey_display_map.clear()
        self.active_jersey_ids.clear()
        self.jersey_position_history.clear()
        self.track_jersey_cache.clear()
        self.jersey_track_cache.clear()
        self.track_to_number.clear()
        self.track_number_conf.clear()
        self.number_history.clear()
        self.occurrences_per_number.clear()
        self.last_ocr_frame.clear()
        self.last_consensus.clear()
        self.pending_number.clear()

    @property
    def display_map(self) -> Dict[str, int]:
        return self.jersey_display_map

    def export_csv(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        track_csv = output_dir / "track_number.csv"
        numbers_csv = output_dir / "numbers_to_tracks.csv"
        with track_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "jersey", "first_frame", "last_frame"])
            for track_id, hist in self.number_history.items():
                if not hist:
                    jersey = None
                    first_frame = None
                    last_frame = None
                else:
                    jersey = hist[-1][1]
                    first_frame = hist[0][0]
                    last_frame = hist[-1][0]
                writer.writerow([track_id, jersey, first_frame, last_frame])
        with numbers_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["jersey", "track_ids"])
            for jersey, ids in self.occurrences_per_number.items():
                writer.writerow([jersey, " ".join(map(str, sorted(ids)))])
