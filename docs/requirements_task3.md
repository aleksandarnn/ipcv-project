# Task 3 requirements: multiplayer detection and identity tracking

**Owner:** Akash Bharadwaj Nanda Kumar (s3579816)
**Game:** Samurai Showdown, two players, one webcam

## What my task does

I detect both players in the frame and give each one a persistent identity — internally `player_id=0` and `player_id=1`, displayed as P1 and P2 everywhere in the UI via `player_label()` from `game_types.py`.

Every other task reads `state.players` to know who is who. Without stable IDs, Task 1 puts the wrong mask on the wrong face, Task 4 credits the wrong player for a hit, and scores break the moment two players cross positions.

My module runs **first** in the pipeline, before face, pose, and control. Identity populates `state.players` with stable IDs and bounding boxes; Task 2 (pose) and Task 1 (face) then attach their results to those existing Player objects.

Design references: **SORT** (Bewley 2016), **HopTrack** (Li et al. 2024).

## What I built

- Person detection on the raw frame using bounding boxes. Three selectable backends via `--detector`: YOLOv8-nano (`yolo`), MediaPipe BlazePose (`mediapipe`), MOG2 background subtraction (`mog2`). Default is YOLO.
- Initial identity assignment: leftmost detected person → P1 (player_id=0), rightmost → P2 (player_id=1).
- Frame-to-frame identity matching using a two-stage pipeline:
  - **Stage 1 (IoU):** Hungarian assignment on predicted bounding box overlap — handles normal movement.
  - **Stage 2 (histogram):** HSV colour histogram fingerprint matching — handles crossing and partial occlusion where IoU alone is ambiguous.
- Kalman filter prediction keeps each track alive between detector calls (detector runs every 3 frames by default).
- Re-entry matching: when a player who left the frame comes back, their histogram fingerprint is compared against all stored tracks and they are restored to their original ID.
- `player_lost` and `player_returned` events fired into `state.events` so Task 4 can pause that player's controls.
- Late-joining player support: if P2 enters the frame 20–30 s after P1, a new track is spawned for them rather than failing silently.
- `DrawCommand`s appended to `state.draw` for a debug bounding box and player label (P1/P2) around each tracked person (layer 2).
- `--metrics` flag prints SORT and HopTrack metrics at exit (see Metrics section).

## Performance targets vs actuals

My budget from `game_types.py` is **3 ms per frame**.

| Target | Actual (YOLO, normal lighting) |
|--------|-------------------------------|
| Module latency ≤ 3 ms/frame | 9.99–24.67 ms/frame (mean 16.55 ms across all scenarios) |
| Detection rate ≥ 95% | 83–100% depending on scenario (100% baseline, 83% temporary exit) |
| Zero ID switches during normal round | 0 baseline; 1 automated switch per crossing session; ~4/10 fast crossings cause silent IoU swaps |
| Re-ID time ≤ 10 frames on re-entry | Mean 8.6 f crossing, 10.0 f occlusion; worst case 107 f (extended absence) |

The latency target is exceeded because YOLOv8-nano detection calls (~17 ms each) dominate the cost. Under baseline conditions the detector runs every 3 frames, bringing the averaged per-frame cost close to budget. A budget renegotiation was noted as likely if detection could not fit in 3 ms.

## Metrics measured and reported

Run `python main.py --metrics` to print at exit. All metrics are printed only when this flag is passed.

| Metric | How measured |
|--------|-------------|
| Module latency (ms/frame) | Self-timed inside `process()` using `time.perf_counter()` |
| Detection rate (%) | Frames each player was actively tracked / total frames |
| ID switches (SORT) | Count of failed re-entry matches; also expressed as /min |
| Track fragments (SORT) | Number of gaps opened in each player's track |
| Centroid jitter px RMS (MOTP proxy) | Welford online variance of tracked centre x,y |
| Re-ID time (frames) | Gap from lost to returned, mean and worst case per player |
| Track continuity (AssA proxy, HopTrack) | Mean and longest continuous run length per player |
| Simultaneous detection rate (HopTrack) | % frames where both P1 and P2 were present simultaneously |
| IDF1 proxy (HopTrack) | re-entry matches / (re-entry matches + ID switches) |
| Hop attempt rate (HopTrack) | Total re-associations / total frames |

### Scenario results — normal indoor lighting (YOLO backend)

| Scenario | Duration | P1 detection | P2 detection | ID switches | Fragments | Mean Re-ID time | Simultaneous det. | Latency |
|----------|----------|-------------|-------------|-------------|-----------|-----------------|-------------------|---------|
| Baseline | 25 s / 300 f | 100.0% | 100.0% | 0 (0.00/min) | 0 | — | 100.0% | 9.99 ms/f |
| Crossing | 141 s / 987 f | 94.1% | 91.2% | 1 (0.43/min) | 16 | P1: 388 ms · P2: 316 ms | 84.9% | 17.95 ms/f |
| Temporary exit | 102 s / 1185 f | 83.0% | 84.4% | 1 (0.59/min) | 18 | P1: 1096 ms · P2: 288 ms | 20.8% | 24.67 ms/f |
| Full occlusion | 42 s / 427 f | 100.0% | 93.0% | 0 (0.00/min) | 3 | P2: 400 ms | 93.0% | 13.59 ms/f |
| **Average** | — | **94.3%** | **92.2%** | **0.5** | **9.25** | **608 ms (34 events)** | **74.7%** | **16.55 ms/f** |

### Scenario results — dim lighting (YOLO backend)

| Scenario | Duration | P1 detection | P2 detection | ID switches | Fragments | Mean Re-ID time | Simultaneous det. | Latency |
|----------|----------|-------------|-------------|-------------|-----------|-----------------|-------------------|---------|
| Baseline | 22 s / 217 f | 100.0% | 98.2% | 0 (0.00/min) | 1 | P2: 120 ms | 73.7% | 9.89 ms/f |
| Crossing | 37 s / 384 f | 78.9% | 73.2% | 0 (0.00/min) | 21 | P1: 404 ms · P2: 312 ms | 54.2% | 16.36 ms/f |
| Temporary exit | 38 s / 395 f | 70.1% | 80.4% | 0 (0.00/min) | 31 | P1: 300 ms · P2: 268 ms | 52.7% | 17.73 ms/f |
| Full occlusion | 26 s / 256 f | 66.4% | 87.1% | 0 (0.00/min) | 14 | P1: 344 ms · P2: 328 ms | 58.6% | 15.74 ms/f |
| **Average** | — | **78.9%** | **84.7%** | **0** | **16.75** | **314 ms (66 events)** | **59.8%** | **14.93 ms/f** |

**Lighting impact summary:** Average P1 detection dropped 15.4 pp and P2 dropped 7.5 pp under dim light. Track fragmentation nearly doubled (37 → 67 total). Zero ID switches in all dim-light runs — not an improvement, but a sign the detector misses players entirely rather than misassigning them. Module latency was unaffected.

## What I write into state

```python
# state.players — one Player per detected person, stable across frames
# player_id is internal (0/1); display uses player_label(player_id) → "P1"/"P2"
state.players = [
    Player(player_id=0, frames_missing=0, colour=(255, 80, 80)),   # P1
    Player(player_id=1, frames_missing=0, colour=(80, 80, 255)),   # P2
]

# state.events — tracking events only; game events belong to Task 4
state.events.append(GameEvent(kind="player_lost",     player_id=X))
state.events.append(GameEvent(kind="player_returned", player_id=X))
```

`frames_missing` is 0 when the player was seen this frame and increments each frame they are not detected. Players remain in `state.players` with `frames_missing > 0` rather than being removed, so Task 4 can see they are temporarily gone without losing their health and score.

## What I need from each teammate

**Task 5 (Aleksandar) — pipeline order.**
Identity must run first. The current order in `main.py` is `identity → face → pose → control`.

**Task 2 (Valerii) — no hard dependency.**
I populate `state.players` with stable IDs and bounding boxes. Task 2 looks up each existing Player by `player_id` and adds keypoints to it. `allow_stubs` is set to `False` on the pose module when identity is loaded, so pose uses identity boxes rather than fixed stub crops.

**Task 4 (Vaishnav) — lost-player behaviour.**
Read `player.frames_missing` before acting on any keypoint signal. If above a threshold (currently `FREEZE_AFTER` in `control.py`) treat that player's controls as frozen. `player_lost` and `player_returned` events gate attack eligibility during re-entry.

## Failure cases

- **Fast crossing (known limitation):** When players cross at speed, the IoU stage reassigns tracks based on position alone and the histogram check is never triggered, causing a silent identity swap. Observed in ~4/10 fast-crossing trials. The automated ID switch counter does not capture this.
- **Extended absence:** If a player is absent long enough for lighting conditions to change, the stored histogram fingerprint diverges from the re-appearing crop. Worst-case Re-ID time observed: 107 frames (4.28 s) under normal light.
- **Dim lighting:** Detection rate drops 8–34 pp depending on scenario. The worst case is P1 during full occlusion (100% → 66.4%). Fragmentation nearly doubles; the tracker loses players briefly and frequently rather than holding tracks. ID switches drop to zero because detections are missed entirely rather than misassigned.
- **Latency over budget:** The 3 ms budget is exceeded whenever YOLO detection runs (once every 3 frames normally, more frequently when tracks are uncertain). This is a known trade-off; MOG2 backend stays near budget but has lower accuracy.
- **Single-player start:** If only one player is present at launch, the second player is detected and assigned a new track when they enter. The histogram for the first player is established correctly; the second player's identity is stable from first detection onward.

## Resolved decisions

- Identity runs **first** in the pipeline (before face, pose, control).
- `frames_missing` threshold for declaring a player truly absent: `MAX_LOST` (defined in `identity.py`).
- Player display names are P1 and P2 (1-indexed) via `player_label()` in `game_types.py`. Internal `player_id` remains 0/1.
- Late-joining second player is supported via new track spawning in `process()`.
