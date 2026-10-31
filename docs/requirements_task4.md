# Task 4 requirements: player interaction and game control

**Owner:** Vaishnav (s number)
**Game:** Samurai Showdown, two players, one webcam
**Code:** `game/modules/control.py`, tests in `game/tests/test_control.py`

## What my task does

I turn keypoints into a game. Pose (task 2) says where the joints are and
identity (task 3) says who is who. I decide what each player is doing, whether
an attack lands, and who wins. I run last of the four modules, after
identity, face and pose, so everything I read is up to date.

1. **Recognise actions** from keypoints: attack, guard, dodge.
2. **Resolve attacks** against what the defender did: `damage`, `blocked` or `missed`.
3. **Keep the score**: health per round, rounds won, KO.
4. **Run the round flow**: waiting for two players, a 3-2-1 countdown, the fight, KO, then the next round.
5. **Raise events** from `game_types.EVENTS` so task 5 can show them.

## How actions are recognised

All distances are in **shoulder widths**, not pixels, so a player near the
camera and one far away need the same movement. If a player turns sideways
their shoulder width shrinks, so the scale also uses torso length.

| Action | Rule | Event |
|---|---|---|
| Attack | A wrist moves faster than 4 shoulder widths/s **relative to the shoulders**, away from the body, at least 0.7 widths from the chest. Cooldown 0.7 s. Not possible while guarding or within 0.3 s of dropping the guard. | `attack`, strength 0.3–1.0 from speed |
| Guard | Wrists crossed (their left/right order is opposite to the shoulders'), at chest height, close together, and in front of the body's middle (within 0.6 widths). Has to hold for 0.4 s to start and 0.2 s to drop. Arms already crossed in time also block a swing before the guard is confirmed. | `block`, every frame while held |
| Duck | Shoulders drop 0.45 widths below the player's own standing height, which is learned while they stand | `dodge`, once when it starts |
| Sidestep | Torso moves 0.5 widths sideways within 0.5 s | `dodge`, once when it starts |

The guard hold time and the sidestep limits were changed after the live
measurement: at about 9 fps a cut's wind-up crossed the arms long enough to
count as a guard (3 frames), and real quick sidesteps only moved 0.2-0.5
widths within 0.3 s.

Wrist speed is measured relative to the shoulders. Walking or ducking moves
the whole body, and that must not count as a swing. The "away from the body"
rule stops a fast guard, which pulls the arms inward, from counting as an
attack. Low-confidence keypoints (`Player.point()` returns `None`) never
trigger anything.

The swing direction is horizontal or vertical, taken from the wrist velocity.

## How attacks are resolved: by timing, not reach

One webcam cannot measure depth, so "did the sword reach" cannot be measured
reliably. Instead, every attack gives the defender **0.35 s** to react. A
defence started up to 0.2 s before the swing also counts.

| Attack | Stopped by | Result |
|---|---|---|
| Vertical cut | guard (crossed arms) | `blocked` |
| Horizontal cut | duck | `missed` |
| Either | sidestep | `missed` |
| No valid defence in time | — | `damage`, 8–20 HP depending on strength |

This follows `docs/game_flow.md`. The table is the `DEFENCES` constant in
`control.py`, so the team can change it in one place.

## Simultaneous and conflicting actions

- **Guard and attack at once:** the guard wins and no attack is possible.
  Dropping the guard also blocks attacks for 0.3 s, because lowering the arms
  fast looks exactly like a swing.
- **Both players swing within 0.15 s:** the blades clash. Both swings are
  cancelled, both players get `blocked`, and a CLASH spark appears between them.
- **Swings further apart than that** are resolved independently, so both
  players can land a hit (a trade).

## Visual effects on the players

The brief says task 4 adds the effects aligned with each player's body. Task 5
reacts with the whole scene (shake, flash, arena). All of these are
`DrawCommand`s, so the "nobody draws on the frame" rule still holds.

| Effect | Where it is anchored | When |
|---|---|---|
| Katana | Along the forearm, elbow → wrist, of the hand that swung last | Always |
| Slash arc | The path the wrist actually took, shrinking over 0.35 s | `attack` |
| Guard shield | A pulsing ring where the wrists cross | While guarding |
| Hit spark + "-13" | On the defender's chest, the number rises and fades | `damage` |
| White spark + "BLOCKED" | On the defender's chest | `blocked` |
| "MISS" | Where the blade was | `missed` |
| Afterimage | Where the body was before the duck or step | `dodge` |
| "CLASH!" | Between the two blades | Clash |
| "K.O." | On the loser | `ko` |
| GUARD / DUCK / STEP / READY? | Above the head | While active |
| Health bars, wins | Top of the frame | Always |

## Trying it without pose and identity

`python tools/demo.py` plays a scripted duel between two stick figures through
the real control module, arena and renderer: hit, block, duck, sidestep,
clash and KO. `--out demo.avi` records it, and `--headless --shots dir/`
saves key frames for the report.

## What I write into state

```python
player.health, player.score   # copied in every frame; score = rounds won
state.phase                   # "waiting" (also during countdown), "playing", "over"
state.message                 # "WAITING FOR 2 PLAYERS", "ROUND 2  3", "FIGHT!", "P0 WINS ROUND 1"
state.events                  # attack, block, dodge, damage, blocked, missed, ko, round_start
state.draw                    # health bars (layer 2), action labels and swing trail (layer 1)
```

The `player_id` on each event:

- `attack`, `block`, `dodge` name the player doing it.
- `damage` and `blocked` name the **defender**, the one who got hit or blocked.
- `missed` names the **attacker**, whose swing hit air.
- `ko` names the loser. `round_start` has `player_id = -1`.

## Why the module keeps its own memory

`main.py` builds a new `GameState` every frame, so health written to a
`Player` is gone on the next frame. I keep health, rounds and motion history
inside the module, keyed by `player_id`, and copy them back each frame. When
`r` is pressed, `reset()` clears everything.

## Lost players

As task 3 suggested, a player with `frames_missing > 5` is **frozen**. They
cannot act, swings aimed at them are cancelled, their motion history is
cleared, and the screen shows `P1 LOST - STEP BACK IN`. When they return they
get a **1 s grace period** before they can attack or be hit, so nobody gets
punished by a tracking glitch. Health is kept while they are gone.

## Performance target

The budget is 3 ms per frame. Measured: **about 0.06 ms per frame** with two
players (the test suite checks this). The module only does arithmetic on a
few keypoints.

## Metrics I will measure and report

| Metric | How |
|---|---|
| Module latency | `profiler.py` summary on exit |
| Action detection rate | Perform each action 20 times per player, count detections (from the event log) |
| False triggers per minute | 2 minutes of standing, talking and walking without fighting; count `attack`/`dodge`/`block` |
| Action latency | Frames from the start of a movement to its event, using a recorded clip |
| Outcome correctness | Scripted duels: does each swing resolve to the intended damage, blocked or missed |
| Robustness | The same tests near and far from the camera (scale invariance) and with one player briefly leaving |

Set `IPCV_CONTROL_LOG=events.csv` before running and every event is written to
a CSV file with timestamp, frame, event, player and strength. That file is the
raw data for the counts above.

## Integration with pose and identity

Pipeline order: identity → face → pose → control.

**From Task 3 (Akash):**
- I use `player_id`, `frames_missing` and `colour`.
- Identity clears `bbox` while a player is lost, so pose skips them and they
  arrive with no keypoints. I freeze them after `frames_missing > 5`.
- A player that has been evicted from `state.players` is treated as lost.

**From Task 2 (Valerii):**
- **Crop-normalised keypoints.** Pose writes keypoints in 0–1 coordinates of
  its own padded, smoothed crop, not of identity's `bbox`. I therefore added
  `Player.crop` (in `game_types.py`), which pose sets to that exact box, and I
  convert back to pixels with `px = crop.x0 + x·crop.w`.
- **Why pixels matter.** The crop follows the player, so in crop units a
  sidestep or a duck barely moves the shoulders, and the actions could not be
  detected. Body-relative units (shoulder widths) are then computed from the
  pixel positions.
- **What I use:** shoulders (reference point), wrists and elbows (attacks,
  guard, katana), hips (scale). There is no `nose` keypoint any more, so labels
  are placed above the shoulders.
- **Guessed joints.** Pose fills hidden joints from a template (confidence
  0.58), from torso-relative memory (0.70), or by prediction (decaying). These
  are fine for drawing, but a wrist may only start an attack at confidence
  ≥ 0.75, which in practice means a real MediaPipe detection.
- **Velocity.** Pose also provides `vx, vy`. I compute my own wrist velocity
  relative to the shoulders, in pixels, so walking does not count as a swing.
- **Round start.** A round only starts when pose has seen both bodies in the
  last second, not just when identity has two boxes.

Tested with a two-person photo through the real modules: the katana lands on
the real hand. Measured on my laptop at 960 px width: pose 50–60 ms, identity
9–11 ms, control 0.2 ms, 11–13 fps overall.

Pose problems seen in that test:
- When two players overlap, pose can put one player's wrist on the other's
  arm, because the crops overlap.
- A leaning player in a wide box sometimes gets no keypoints at all.

**Task 5 (Aleksandar).**
- I draw health bars as `DrawCommand`s on layer 2, in the top 12 % of the frame
  where your arena beam is. If you would rather draw the HUD yourself from
  `player.health`, set `DRAW_HUD = False` in `control.py`.
- `block` arrives every frame while the guard is held, so the glow can stay on.
- I use `rect` with `thickness=-1` for filled bars, which `cv2.rectangle` supports.

## Not decided yet

- Whether guard should also stop horizontal cuts (now it does not, as in `game_flow.md`).
- Thresholds are first guesses from synthetic tests. They now need tuning on the live camera with two people.
- Jump dodge: pose now provides knees and ankles, so it can be added.
- The finger-rune shield from `game_flow.md`: the finger names exist, but pose does not fill them yet.
