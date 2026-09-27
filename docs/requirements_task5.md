# Task 5 requirements: scene processing and real-time integration

**Owner:** Aleksandar Nikolov (s2523477)
**Game:** Samurai Showdown, two players, one webcam
**Last updated:** 27 September 2026

This document has two halves. The first says what my own task must deliver.
The second says what I need from tasks 1 to 4, which the project brief makes
my responsibility as team lead.

Every requirement has an ID so we can point at it in meetings and in the
report, and a test so we can tell whether it is met rather than argue about it.

---

## 1. Scope

I own the application that everything else plugs into: the camera, the main
loop, the order modules run in, the timing, the failure handling, the scene
and its reactions, and the final integrated build.

I do **not** own face detection, pose estimation, identity assignment, or the
rules of the duel. Those are tasks 1 to 4.

---

## 2. Functional requirements

| ID | Requirement | How it is tested |
|---|---|---|
| F1 | One executable application starts the game from a single command, with no manual setup beyond installing the requirements file. | Clone the repository on a second machine, install, run, observe the game. |
| F2 | The loop captures webcam frames, runs all four task modules in the fixed order face, pose, identity, control, then renders. | Order is visible in `main.py`; timing panel lists all five stages. |
| F3 | Task modules are discovered automatically from `game/modules/`. A task whose file is absent is skipped and reported, not crashed on. | Delete a module file, run, observe it listed as missing and the game still running. |
| F4 | The scene renders an arena background and a heads-up display showing each player's health, name and the round state. | Visible on screen with two players present. |
| F5 | The scene reacts visibly and differently to `damage`, `blocked` and `missed`, so an observer can tell the three apart without being told. | Trigger each event; three different reactions. |
| F6 | The scene reacts to `ko` and `round_start` with a clear state change. | Play a round to a knockout. |
| F7 | All drawing happens in the renderer. Modules submit draw commands and never write to the frame. | Code review: no module imports drawing functions. |
| F8 | A single key hides the performance panel so the demo can be shown clean. | Press `s` during the demo. |

---

## 3. Performance requirements

Targets are for the demonstration machine with two players in frame, at the
resolution we ship.

| ID | Requirement | Target | How it is measured |
|---|---|---|---|
| P1 | Sustained frame rate during normal play | at least 20 fps, aiming for 25 | rolling average over 60 frames, printed on exit |
| P2 | End-to-end latency, camera frame to displayed frame | 100 ms or less, median | measured per frame in `profiler.py` |
| P3 | Per-module time budgets are respected | face 8 ms, pose 15 ms, identity 3 ms, control 3 ms, render 6 ms | per-module timing panel |
| P4 | The bottleneck is identified and named in the report, with evidence | one module named, with its measured cost | timing summary before and after optimisation |
| P5 | At least one measured optimisation is reported, with the frame rate before and after | any real improvement | two timing summaries |

If a module cannot meet its budget, the budget is renegotiated in a meeting
rather than silently exceeded, because the total is what the player feels.

---

## 4. Robustness requirements

The brief asks for controlled behaviour when things go wrong. These are the
cases I handle.

| ID | Situation | Required behaviour | How it is tested |
|---|---|---|---|
| R1 | A module raises an exception | The frame still renders, the failure is counted and printed. | Insert a module that always raises. |
| R2 | A module fails repeatedly | After 5 failures it is disabled, shown as OFFLINE, and the game continues. | Same test, run for more than 5 frames. |
| R3 | A player leaves the frame | The game pauses or holds state rather than acting on missing data. No attack is registered for a player who is not visible. | Walk out of frame mid-round. |
| R4 | A player returns | Their identity, health and score are the same as before they left. | Walk out and back in. |
| R5 | Only one player is visible at the start | The game waits and says so, rather than starting a duel with one player. | Start with one person in frame. |
| R6 | The camera cannot be opened | The program exits immediately with a clear message naming the problem. | Run with the camera in use by another application. |
| R7 | A three-minute demonstration runs without a crash | No unhandled exception, no freeze. | Full rehearsal, timed. |

---

## 5. What I need from tasks 1 to 4

These are the interfaces. All types are defined in `game/game_types.py`.

### From task 1, face tracking

| ID | Requirement |
|---|---|
| I1.1 | A `Face` for each visible player: centre, width, height, and confidence, in pixels of the frame I passed in. |
| I1.2 | Faces attached to the correct `Player`, not returned as a loose list. |
| I1.3 | Face effects submitted as draw commands on layer 2, never drawn onto the frame. |
| I1.4 | When a face is not found, leave `player.face` as `None` rather than guessing a position. |
| I1.5 | Runs within 8 ms per frame for two players. |

### From task 2, pose estimation

| ID | Requirement |
|---|---|
| I2.1 | Keypoints written into `player.keypoints`, using exactly the names in `KEYPOINTS`. |
| I2.2 | Every keypoint carries an honest confidence. Do not report 1.0 for a guess. |
| I2.3 | Smoothing applied inside the module, so what I receive is already stable. |
| I2.4 | A missing keypoint is reported with low confidence, never as a stale or invented position. |
| I2.5 | Hand speed made available, since attack strength depends on it. Agree the field name with me before adding it. |
| I2.6 | Runs within 15 ms per frame for two players. |

### From task 3, identity

| ID | Requirement |
|---|---|
| I3.1 | `player_id` stays with the same human across the whole round, including crossing, occlusion, leaving and re-entering. |
| I3.2 | `frames_missing` maintained: 0 when seen this frame, incrementing otherwise. |
| I3.3 | `player_lost` and `player_returned` events raised, so the scene can react. |
| I3.4 | Exactly two players in the list during a duel. No duplicates, no ghosts. |
| I3.5 | Runs within 3 ms per frame. |

### From task 4, game control

| ID | Requirement |
|---|---|
| I4.1 | Only the events listed in `game_types.py` are raised, spelled exactly. New ones are agreed with me first. |
| I4.2 | `damage` names the player who was **hit**, not the one who swung. |
| I4.3 | `player.health` updated for the correct player. |
| I4.4 | Cooldowns or state logic prevent one swing registering as many attacks. |
| I4.5 | No action is registered for a player whose keypoints are unreliable or who is not visible. |
| I4.6 | Runs within 3 ms per frame. |

### What I provide to everyone

| ID | Provision |
|---|---|
| O1 | A frame, already resized and mirrored, once per loop. Nobody else opens the camera. |
| O2 | A fresh `GameState` each frame, with players carried over from the previous frame. |
| O3 | Rendering of every submitted draw command, in layer order. |
| O4 | Timing of every module, reported live and on exit. |
| O5 | Protection from crashes: an exception in one module does not stop the game. |

---

## 6. Assumptions and constraints

Both players are in one frame, waist up, side by side. A single webcam, no
other hardware, as the brief requires. Depth is not directly measurable, so
anything depending on distance from the camera has to be inferred from
apparent size, and needs a per-player baseline at the start of a round.

---

## 7. Out of scope for me

Detection accuracy of faces, keypoints or identities. The rules, balance and
feel of the duel. Those belong to their task owners, and their own
requirements documents should cover them.
