# Task 1 requirements: face tracking and augmented effects

**Owner:** Shilpa John (s3864693)
**Game:** Samurai Showdown, two players, one webcam

## What my task does

I detect the face of every player, keep it smooth and stable over time, and draw
a samurai headband on each player's forehead that follows the head when the
player moves, tilts or comes closer to the camera. Above each head I draw a
health bar with the player's name ("P0 85"), so everyone can see who is who and
how much health they have left.

Every face is attached to the player identity from Task 3. Without that, the
headband and the health bar could end up on the wrong person when the players
cross.

My module runs second in the pipeline: identity → **face** → pose → control → render.

## What I will build

- Face detection on the full frame with MediaPipe BlazeFace (full-range model, for players 1.5–3 m from a laptop webcam).
- Face position, size and head tilt (from the line between the eyes), plus a rough left/right turn (from the nose position).
- Matching of each face to a Task 3 player: the face must be inside that player's body box (with some room above it, because the box can start at eye level) and close to where that face was predicted.
- Temporal smoothing against jitter: a Kalman filter for the face centre, an exponential moving average for size and tilt.
- Controlled behaviour when a face is briefly lost: the Kalman prediction keeps the headband in place for up to 5 frames, then it is removed; it comes back when the face is detected again.
- The face effect: a cloth headband in the player's colour, drawn in face coordinates so it scales and rotates with the head.
- The extra effect (not on the face): a health bar and name label above each head, showing the health kept by Task 4.

## Performance target

My budget from `game_types.py` is **8 ms per frame**. I am aiming for:

- Average module time ≤ 8 ms at 960 px width with two players.
- Zero effect switches between players during a normal round, including when players cross.
- Match rate ≥ 95 % of detected faces attached to a player.
- Less jitter after filtering than in the raw detections while a player stands still.

## Metrics I will measure and report

| Metric | How |
|--------|-----|
| Module time (ms/frame) | `profiler.py` summary at exit (press `q`, or `--headless 600`), 1-player and 2-player |
| Effect switches | Count in a recorded two-player clip with crossings: how often a headband jumps to the other person |
| Match rate (%) | Counted in `face.py` (matched faces / detected faces), printed when the game exits |
| Jitter (px) | Logged in `face.py`: mean frame-to-frame movement of the face centre while a player stands still, raw detection vs Kalman-filtered |

| Scenario | What I measure |
|----------|----------------|
| Both players standing still, about 20 s | Jitter (raw vs filtered) |
| Normal play: moving, crossing, coming closer and going back | Match rate, module time, effect switches |
| Hand briefly covering the face | Short detection failures: headband stays briefly, then returns on the right player (shown in the demo video, no number) |

The code that counts the match rate, logs the jitter and prints both at the end of a run will be added to `face.py` before the evaluation.

## What I write into state

```python
# player.face: only when the face was really detected this frame, else None
player.face = Face(cx, cy, w, h, angle, confidence)

# state.draw: headband (layer 1) and health bar + label (layer 2)
state.draw.append(DrawCommand(kind="line", ...))     # headband strokes
state.draw.append(DrawCommand(kind="rect", ..., layer=2))   # health bar
state.draw.append(DrawCommand(kind="text", text="P0  85", layer=2))
```

I never write a predicted (guessed) face into `player.face`, so other tasks do not
treat predictions as actual face detections

With `--debug` I also draw the face box: green = detected this frame, cyan =
predicted.

## What I read from state

- `state.players` from Task 3: `player_id`, `bbox`, `frames_missing`.
- `player.health` from Task 4. Control runs after face, so I read the health
  from the previous frame's players (one frame, about 33 ms, late).

## What I need from each teammate

**Task 3 (Akash) — player boxes.**
I only attach a face to a player that has a body box. Please make sure the box
includes the head, keep the same `player_id` for the same person all round,
and also detect a player who only arrives after the game has started. If a
player has no box, they get no headband or health bar, even when their face is
visible.

**Task 4 (Vaishnav) — health.**
Identity creates new `Player` objects every frame, so keep each player's health
in your module by `player_id` and write it into `player.health` every frame, in
the range 0–100. Otherwise the health bar stays at 100.

**Task 5 (Aleksandar) — drawing**
My effects are only draw commands, the renderer draws them.

**Task 2 (Valerii) — no dependency.**

## Failure cases

- No identity box for a player no headband or health bar for that player.
- Face hidden for more than 5 frames: the headband disappears until the face is seen again.
