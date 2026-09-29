# Task 3 requirements: multiplayer detection and identity tracking

**Owner:** Akash (s number)
**Game:** Samurai Showdown, two players, one webcam

## What my task does

I detect both players in the frame and give each one a persistent identity — Player 0 and Player 1.
Every other task reads `state.players` to know who is who. Without stable IDs, Task 1 can put the
wrong mask on the wrong face, Task 4 credits the wrong player for a hit, and scores break the moment two players cross positions.

My module runs third in the pipeline, after face and pose, before control.

But in game/readme task3 has to run at first????

## What I will build

- Person detection on the raw frame using bounding boxes (no dependency on Task 2 or else use the keypoints/signals from Task2).
- Initial identity assignment: leftmost detected person becomes Player 0, rightmost becomes Player 1.
- Frame-to-frame identity matching so the labels follow the correct person even when players move.
- Player prediction to keep the identies remain same so that when players cross over identities are not changed
- Controlled re-entry: when a player who left the frame comes back, they are assigned back to their original ID. Keep track of identies and where they are on the spatial region
- `player_lost` and `player_returned` events fired into `state.events` so Task 4 can pause that player's controls and Task 5 can show something on screen.
- A time-budget-aware path: detection is the expensive step, matching is cheap

## Performance target

My budget from `game_types.py` is **3 ms per frame**. I am aiming for:

- Average module latency ≤ 3 ms at 480 × 270 resolution (default `--width 480`), two players.
- Detection rate ≥ 95 % of frames under normal indoor lighting.
- Identity switch rate: zero switches during a normal round; measured across a crossing scenario in testing.
- Re-identification time after a player re-enters: ≤ 10 frames (≈ 0.4 s at 25 fps).

If detection alone cannot fit in 3 ms I will discuss a budget adjustment.

## Metrics I will measure and report

| Metric | How |
|--------|-----|
| Module latency (ms/frame) | `profiler.py` output, collected under 1-player and 2-player conditions |
| Identity switch count | Manual ground-truth annotation of a 30 s crossing clip |
| Correct ID rate (%) | Labelled frames where both IDs are correct / total frames |
| Re-ID time (frames) | Timed across ≥ 5 player-exit-and-return trials, mean and worst case |
| Bounding box centroid jitter | Variance of tracked centre x,y while player stands still |
| Detection rate (%) | Frames where both players are detected / total frames |

I will run each of these in at least three scenarios: standing still, repeated
crossing, and one player leaving and returning. The raw numbers go in the report.

## What I write into state

```python
# state.players — one Player per detected person, stable across frames
state.players = [
    Player(player_id=0, frames_missing=0, colour=(255, 80, 80)),   # blue, P0
    Player(player_id=1, frames_missing=0, colour=(80, 80, 255)),   # red,  P1
]

# state.events — tracking events only; game events belong to Task 4
state.events.append(GameEvent(kind="player_lost",     player_id=X))
state.events.append(GameEvent(kind="player_returned", player_id=X))
```

`frames_missing` is 0 when the player was seen this frame, and increments each
frame they are not detected. I keep them in `state.players` with `frames_missing > 0`
rather than removing them, so Task 4 can see they are temporarily gone without
losing their health and score.

I also append `DrawCommand`s to `state.draw` for a debug bounding box and player
label around each tracked person (layer 1, toggled off by default in release mode).

## What I need from each teammate

**Task 5 (Aleksandar) — pipeline order.**
The current order in `main.py` is face → pose → identity → control

**Task 2 (Valerii) — no hard dependency.**
I populate `state.players` with stable IDs. If the order is changed so I run first, Task 2 looks up the existing Player by `player_id` and adds keypoints to it.

**Task 4 (Vaishnav) — lost-player behaviour.**
Read `player.frames_missing` before acting on any keypoint signal. If it is above a threshold (suggest 5 frames) treat that player's controls as frozen. I also give `player_lost` and `player_returned` events you can use to gate attack eligibility during re-entry.

## Failure Cases

- 

## Not decided yet

- Whether I run before or after Task 2 — needs team agreement with Aleksandar and Valerii.
- The exact `frames_missing` threshold at which a player is declared truly absent
