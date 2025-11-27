import math
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Deque, Dict, Optional, Set, Tuple

from JerseyNumber.ocr.jersey_recognizer import JerseyRecogniser

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

    def _build_recogniser(self) -> Optional[JerseyRecogniser]:
        tracker_device = getattr(self.tracker, "device", "cuda:0")
        jersey_device = "cuda" if str(tracker_device).startswith("cuda") else "cpu"
        return JerseyRecogniser(
            parseq_root=Path("/home/ec2-user/workspace/footballp_tracker"),
            checkpoint_path=self.config.parseq_checkpoint,
            pose_model_path=self.config.pose_model_path,
            device=jersey_device,
            history_window=30,
            confidence_threshold=0.6,
            vote_min_confidence=0.55,
            vote_min_support=2,
            vote_high_threshold=0.65,
            vote_count_min=4,
            vote_count_margin=2,
        )

    def process(self, frame, cur_tracks, frame_idx: int, timings, counts):
        if self.jersey_ocr is None:
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))
            self.active_jersey_ids = set(cur_tracks['players'].keys())
            return cur_tracks

        t0 = perf_counter()
        if cur_tracks['players']:
            player_entries = list(cur_tracks['players'].items())
            jersey_candidates: Dict[str, list] = {}
            pending_players = []
            for display_id, info in player_entries:
                bbox = info.get('bbox')
                source_tid = info.get('source_track_id')
                decision = None
                center = None
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                    attempt = (
                        (self.config.ocr_frame_stride <= 1)
                        or (frame_idx % self.config.ocr_frame_stride == 0)
                        or ('jersey_number' not in info)
                    )
                    if attempt:
                        reading = self.jersey_ocr.read_number(frame, (x1, y1, x2, y2), frame_idx=frame_idx)
                        decision = self.jersey_ocr.confirm_number(display_id, reading, frame_idx=frame_idx)
                    else:
                        decision = self.jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                else:
                    decision = self.jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                info.pop('jersey_number', None)
                info.pop('jersey_confidence', None)

                resolved = None
                if (
                    decision
                    and decision.is_confirmed
                    and decision.text
                    and (decision.mean_confidence or 0.0) >= self.config.jersey_cache_min_confidence
                ):
                    resolved = {
                        "jersey": decision.text,
                        "confidence": decision.mean_confidence,
                        "consensus": decision.consensus,
                        "votes": decision.votes,
                        "source": "ocr",
                    }
                elif source_tid is not None:
                    cached = self.track_jersey_cache.get(source_tid)
                    if cached:
                        age = frame_idx - cached.get("frame", -1)
                        jersey = cached.get("jersey")
                        owner = self.jersey_track_cache.get(jersey) if jersey else None
                        owner_active = (
                            owner
                            and owner.get("track_id") is not None
                            and owner["track_id"] != source_tid
                            and frame_idx - owner.get("frame", -1) <= self.config.jersey_cache_ttl_frames
                        )
                        if (
                            jersey
                            and age <= self.config.jersey_cache_ttl_frames
                            and not owner_active
                        ):
                            resolved = {
                                "jersey": jersey,
                                "confidence": cached.get("confidence", 0.5),
                                "consensus": cached.get("consensus", 0.0),
                                "votes": cached.get("votes", 0),
                                "source": "cache",
                            }

                if resolved:
                    jersey = resolved["jersey"]
                    jersey_candidates.setdefault(jersey, []).append({
                        "info": info,
                        "confidence": resolved["confidence"],
                        "consensus": resolved.get("consensus", 0.0),
                        "votes": resolved.get("votes", 0),
                        "center": center,
                        "source_tid": source_tid,
                        "original_id": display_id,
                        "jersey": jersey,
                    })
                    if source_tid is not None:
                        self.track_jersey_cache[source_tid] = {
                            "jersey": jersey,
                            "frame": frame_idx,
                            "confidence": resolved["confidence"],
                            "consensus": resolved.get("consensus", 0.0),
                            "votes": resolved.get("votes", 0),
                        }
                        self.jersey_track_cache[jersey] = {
                            "track_id": source_tid,
                            "frame": frame_idx,
                        }
                else:
                    pending_players.append((info, source_tid, display_id))

            selected_candidates = []
            for jersey, candidates in jersey_candidates.items():
                history = self.jersey_position_history.get(jersey)
                expected_pos = None
                loyalty_display = self.jersey_display_map.get(jersey)
                if history:
                    last_frame, last_cx, last_cy = history[-1]
                    if frame_idx - last_frame <= 20:
                        expected_pos = (last_cx, last_cy)
                        if len(history) >= 2:
                            prev_frame, prev_cx, prev_cy = history[-2]
                            frame_delta = max(1, last_frame - prev_frame)
                            vx = (last_cx - prev_cx) / frame_delta
                            vy = (last_cy - prev_cy) / frame_delta
                            delta_frames = max(0, frame_idx - last_frame)
                            expected_pos = (last_cx + vx * delta_frames, last_cy + vy * delta_frames)

                def candidate_key(candidate):
                    center = candidate['center']
                    loyalty_rank = 0 if loyalty_display == candidate['original_id'] else 1
                    if center is None:
                        distance = 1e9
                    else:
                        if expected_pos is not None:
                            distance = math.hypot(center[0] - expected_pos[0], center[1] - expected_pos[1])
                        elif history:
                            _, last_cx, last_cy = history[-1]
                            distance = math.hypot(center[0] - last_cx, center[1] - last_cy)
                        else:
                            distance = 1e9
                    return (
                        loyalty_rank,
                        distance,
                        -candidate['consensus'],
                        -candidate['votes'],
                        -candidate['confidence'],
                    )

                best_candidate = min(candidates, key=candidate_key)
                best_info = best_candidate['info']
                best_info['jersey_number'] = best_candidate['jersey']
                best_info['jersey_confidence'] = best_candidate['confidence']
                selected_candidates.append(best_candidate)
                for other in candidates:
                    if other is best_candidate:
                        continue
                    pending_players.append((other['info'], other['source_tid'], other['original_id']))

            jersey_confirmations = [
                (cand['jersey'], cand['info'], cand['source_tid'], cand['original_id'])
                for cand in selected_candidates
            ]

            assigned_players = {}
            new_player_id_map = {}
            used_ids = set()
            if self.tracker.max_player_ids:
                all_ids = list(range(1, self.tracker.max_player_ids + 1))
            else:
                all_ids = sorted({pid for pid, _ in player_entries})
            reserved_ids = set(self.jersey_display_map.values())
            display_to_jersey_local = {display_id: jersey for jersey, display_id in self.jersey_display_map.items()}

            for jersey, info, source_tid, original_id in jersey_confirmations:
                if jersey not in self.jersey_display_map:
                    available_for_new = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                    if not available_for_new:
                        continue
                    assigned_display = available_for_new[0]
                    self.jersey_display_map[jersey] = assigned_display
                    reserved_ids.add(assigned_display)
                else:
                    assigned_display = self.jersey_display_map[jersey]
                if assigned_display in assigned_players:
                    alternatives = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                    if alternatives:
                        assigned_display = alternatives[0]
                        self.jersey_display_map[jersey] = assigned_display
                        reserved_ids.add(assigned_display)
                    else:
                        continue
                previous_jersey = display_to_jersey_local.get(assigned_display)
                if previous_jersey and previous_jersey != jersey:
                    self.jersey_display_map.pop(previous_jersey, None)
                    display_to_jersey_local.pop(assigned_display, None)
                display_to_jersey_local[assigned_display] = jersey
                used_ids.add(assigned_display)
                info['display_id'] = assigned_display
                if source_tid is not None:
                    new_player_id_map[source_tid] = assigned_display
                assigned_players[assigned_display] = info

            temp_pool = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
            temp_pool.sort(reverse=True)
            for info, source_tid, original_id in pending_players:
                if original_id is not None and original_id not in used_ids:
                    assigned_display = original_id
                elif temp_pool:
                    assigned_display = temp_pool.pop(0)
                else:
                    remaining = [i for i in all_ids if i not in used_ids]
                    if not remaining:
                        continue
                    assigned_display = remaining[0]
                used_ids.add(assigned_display)
                info['display_id'] = assigned_display
                if source_tid is not None:
                    new_player_id_map[source_tid] = assigned_display
                assigned_players[assigned_display] = info

            cur_tracks['players'] = dict(sorted(assigned_players.items()))
            active_ids = set(cur_tracks['players'].keys())
            stale_ids = self.active_jersey_ids - active_ids
            for stale_id in stale_ids:
                self.jersey_ocr.reset_history(stale_id)
            self.active_jersey_ids = active_ids

            self.tracker.sync_display_assignments(self.jersey_display_map, new_player_id_map)

            active_source_tracks = {
                info.get("source_track_id")
                for info in cur_tracks['players'].values()
                if info.get("source_track_id") is not None
            }
            for track_id in list(self.track_jersey_cache.keys()):
                meta = self.track_jersey_cache.get(track_id) or {}
                last_frame = meta.get("frame", -1)
                if (
                    track_id not in active_source_tracks
                    and frame_idx - last_frame > self.config.jersey_cache_ttl_frames
                ):
                    self.track_jersey_cache.pop(track_id, None)
            for jersey_key in list(self.jersey_track_cache.keys()):
                jersey_meta = self.jersey_track_cache.get(jersey_key) or {}
                if frame_idx - jersey_meta.get("frame", -1) > self.config.jersey_cache_ttl_frames:
                    self.jersey_track_cache.pop(jersey_key, None)

            for display_id, info in cur_tracks['players'].items():
                jersey = info.get('jersey_number')
                bbox = info.get('bbox')
                if jersey and bbox:
                    x1, y1, x2, y2 = map(float, bbox)
                    center_x = 0.5 * (x1 + x2)
                    center_y = 0.5 * (y1 + y2)
                    history = self.jersey_position_history.get(jersey)
                    if history is None:
                        history = deque(maxlen=20)
                        self.jersey_position_history[jersey] = history
                    history.append((frame_idx, center_x, center_y))
        else:
            if self.active_jersey_ids:
                for stale_id in self.active_jersey_ids:
                    self.jersey_ocr.reset_history(stale_id)
                self.active_jersey_ids = set()
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))

        timings["jersey_ocr"] += perf_counter() - t0
        counts["jersey_ocr"] += 1
        return cur_tracks

    def reset(self):
        self.jersey_display_map.clear()
        self.active_jersey_ids.clear()
        self.jersey_position_history.clear()
        self.track_jersey_cache.clear()
        self.jersey_track_cache.clear()

    @property
    def display_map(self) -> Dict[str, int]:
        return self.jersey_display_map
