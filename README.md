# IPCV 2026-1A group project

Webcam-based interactive multiplayer game, University of Twente.

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

The game runs today with placeholder modules, so you can see the whole pipeline
working before any real tracking exists. See `game/README.md` for how to plug
your own module in.

## Layout

```
game/                the application
  game_types.py      shared data types every module uses
  main.py            the loop: camera, modules, timing, failure handling
  renderer.py        scene rendering
  profiler.py        frame rate and latency measurement
  modules/           task modules; stubs.py holds the placeholders
```
