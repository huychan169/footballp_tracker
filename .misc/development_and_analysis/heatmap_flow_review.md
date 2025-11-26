# 2D Heat Map Flow Review (updated)

## Flow overview (current code)
- **Tracking & jersey attribution** – `main.py` collects tracks from `Tracker.get_object_tracks`, re-numbers after OCR, and tags teams (`main.py:100-360`).
- **Homography estimation** – Each frame calls `vt.update_homography_from_frame`, which runs `CamCalib.__call__`; `H` is normalized and blended with the previous one, and `last_good_H` is kept (`FieldMarkings/run.py:25-88`, `view_transformer/view_transformer.py:38-42`).
- **Feet projection** – Bboxes are normalized and the bottom is lifted by `feet_offset_ratio` (default 0.08) before unprojection; results are clipped to the minimap plane (`view_transformer/view_transformer.py:62-78`).
- **Camera movement** – Optical-flow dx/dy is only visualized; no extra stabilization is applied to pitch coordinates (`main.py:415-423`).
- **Rendering & cleanup** – `Minimap2D` smooths positions (EMA + max-step) and prunes players by TTL (default 45 frames) before drawing circles; ball keeps a short history (`view_transformer/minimap.py:8-130`). The minimap is blended onto the frame; jersey labels and a stable jersey→track legend overlay are drawn (`main.py:25-75`, `main.py:412-447`).

## Remaining issues impacting placement

### Homography validation still missing
- `CamCalib.__call__` smooths and stores `last_good_H` but accepts any finite `H` without a reprojection quality check (`FieldMarkings/run.py:45-88`). A bad HRNet output can still shift all dots.
- There is no HUD/status indicating when `H` is invalid; `compute_feet` still runs if `H` exists.

**Next steps**
1. Compute a calibration quality score (RMSE/inliers) from `CameraCreator`; skip/keep previous `H` when bad and expose `has_valid_h`.
2. Gate `compute_feet/update_history` on that flag and render a small on-screen HUD (H valid/invalid, last RMSE).

### Feet point is still approximate
- Fixed 8% bbox-height lift helps but is not data-driven; shadows/foreshortening still bias feet inward.
- Outlier bboxes are not filtered before projection.

**Next steps**
1. Use pose ankles when available; otherwise adapt the offset by bbox aspect/height and camera tilt.
2. Reject improbable boxes (too tall/short, extreme aspect) before unprojection.

## What was fixed
- **cam_off drift removed** – Optical-flow offsets no longer shift top-view coords (`main.py:415-423`).
- **Foot offset added** – Bbox bottom is lifted before projection (`view_transformer/view_transformer.py:62-78`).
- **Minimap TTL** – History stores last_seen and prunes stale players, eliminating ghost dots (`view_transformer/minimap.py:8-130`; fed with `frame_idx` in `main.py:431-436`).
- **Overlays** – Jersey numbers over heads and a stable jersey→track legend for monitoring (`main.py:25-75`, `main.py:412-447`).

## Next directions (priority)
1. Homography quality gate + HUD; pause minimap ingestion when `H` is invalid.
2. Better feet localization using pose ankles or adaptive offset and bbox filtering.
3. Minimap gating: if `has_valid_h` stays False for N frames, freeze updates (keep last minimap) until a good `H` returns.
4. Optional fallback stabilization only when `H` is invalid: shift bboxes in image space with bounded optical flow before projection (never after).***
