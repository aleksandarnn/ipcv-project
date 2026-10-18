# Task 2 — Pose module I/O

**Module:** `game/modules/pose.py`  
**Owner:** Valerii

## Input

From Task 3, via `state.players` (one entry per person):

| Field | Type | Meaning |
|---|---|---|
| `player_id` | `int` | Stable ID (0 / 1, …) |
| `bbox` | `BBox` | Person crop in **full-frame pixels** (`x0, y0, x1, y1`) |

Pose runs **once per box** (own MediaPipe VIDEO tracker per `player_id`).

Until Task 3 supplies boxes, the module uses fixed left/right stub crops (`FIXED_BOXES`) and writes `player.bbox` itself for testing.

## Output

Written onto the same `Player` objects:

### `player.keypoints` — `dict[str, Keypoint]`

One entry per joint that is detected, held, predicted, or filled by template/memory. Names are from `KEYPOINTS` in `game_types.py`. **Body joints only** for now; finger names exist in the shared list but are not filled.

| Field | Unit | Meaning |
|---|---|---|
| `x`, `y` | crop-normalised ≈ \[0, 1\] | Position relative to that player’s `bbox` |
| `vx`, `vy` | crop-widths / heights per second | Velocity in the same normalised space |
| `confidence` | 0–1 | Trust; use `kp.ok` (≥ 0.4) or `player.point(name)` |

```text
state.players[i].keypoints = {
    "left_wrist":  Keypoint(x, y, confidence, vx, vy),
    "right_wrist": Keypoint(...),
    "left_shoulder": ...,
    ...
}
```

### Also updated

| Field | Meaning |
|---|---|
| `frames_missing` | Pose miss streak (0 = seen this frame). Task 3 may own this long-term. |
| `bbox` | Set from stub crops if Task 3 left it empty |

## Shape (conceptual)

```text
players[player_id].keypoints[joint_name] → { pos: (x, y), vel: (vx, vy), conf }
```

Lookup helpers: `state.player(0)`, `player.point("left_wrist")`.

## Behaviour (what Task 4 should expect)

- **Smoothing:** One-Euro on positions; velocities use a short lookback window + EMA (not raw one-frame Δpos).
- **Velocity outliers:** if crop-normalised speed &gt; `VEL_OUTLIER_MAX` (6.0), returned `vx, vy` are set to **0**.
- **Occlusion / leave:** missing joints can be **held**, **constant-velocity predicted**, or placed from **torso-relative memory** (last good limb pose attached to the live shoulder/hip).
- **Profile (side view):** when the body collapses to a thin torso line, missing limbs/shoulders are **filled from an anatomical template** (and memory if available). Real MediaPipe detections always win over the template.
- **Re-acquire:** after the person is lost then found again, velocities stay at 0 until tracking locks again (avoids false attacks).
- Estimated / held joints still appear in the dict with mid confidence — filter with `player.point(name)` or `kp.ok` before using them for gameplay.

## Debug overlay

```bash
cd game
python main.py --debug
python main.py --debug --source clip.mp4
python main.py --debug --width 960   # default width is already 960
```

Off by default (`Module.debug = False`). With `--debug`:

| Visual | Meaning |
|---|---|
| Blue / red boxes | Player 0 / 1 crops (`P0`, `P1`; may show `lock`, `miss`, `BACK`, `PROFILE`) |
| Green / blue / red lines | Torso / left limbs / right limbs |
| Dark green torso | Model thinks **back** faces the camera (L/R limb colours also swap) |
| Joint dots | Tracking tier: cyan = fresh, light blue = predicted, magenta = relative/template, orange = held |
| Arrows | Velocity; same colour as the dot; only if speed ≥ threshold |

Production drawing belongs to the renderer.

## Runtime / models

- Prefers `models/pose_landmarker_lite.task`, falls back to `pose_landmarker_full.task`.
- Each crop is downscaled for inference (`INFER_MAX_SIDE`, currently 448).

## Notes for Task 4 / 5

- Prefer `player.point("left_wrist")` so low-confidence or estimated joints can be ignored when needed.
- Attack strength: `player.point("right_wrist").speed` (normalised units; already outlier-filtered).
- Do **not** invent joints in control — pose may already have filled them; decide from `confidence` / `.ok`.
