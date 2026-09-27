# Samurai Showdown

IPCV 2026-1A group project, University of Twente.

A two-player webcam duel. Attack by swinging a hand, defend by crossing both
arms, dodge by moving your body out of the way. The controller is computer
vision: one webcam, two players, no hardware.

## Task owners

| Task | What it covers | Owner |
|---|---|---|
| 1 | Face tracking and augmented effects | TBD |
| 2 | Body pose estimation and motion tracking | TBD |
| 3 | Multiplayer detection and identity tracking | TBD |
| 4 | Player interaction and game control | TBD |
| 5 | Scene processing and real-time integration | Aleksandar Nikolov (s2523477) |

## Getting started

```
pip install -r game/requirements.txt
cd game
python main.py
```

Task modules are loaded automatically from `game/modules/`. Any task whose
file does not exist yet is skipped, so the game always runs and the panel on
screen lists what is still missing. See `game/README.md` for how to add yours.

## Layout

```
game/                the application
  game_types.py      shared data types every module uses
  main.py            the loop: camera, modules, timing, failure handling
  renderer.py        scene rendering
  profiler.py        frame rate and latency measurement
  modules/           one file per task: face.py, pose.py, identity.py, control.py
```
