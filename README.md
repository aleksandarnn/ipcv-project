# Samurai Showdown

IPCV 2026-1A group project, Group 20, University of Twente.

A two-player webcam duel. Attack by swinging a hand, defend by crossing both
arms, dodge by moving your body out of the way. The controller is computer
vision: one webcam, two players, no hardware.

## Task owners

| Task | What it covers | Owner |
|---|---|---|
| 1 | Face tracking and augmented effects | Shilpa John (s3864693) |
| 2 | Body pose estimation and motion tracking | Valerii Fomenko (s3106187) |
| 3 | Multiplayer detection and identity tracking | Akash Bharadwaj Nanda Kumar (s3579816) |
| 4 | Player interaction and game control | Vaishnav Rengan Venkatram (s3594769) |
| 5 | Scene processing and real-time integration | Aleksandar Nikolov (s2523477) |

## Getting started

You need a webcam and Python 3.13. From the project folder:

```
pip install -r game/requirements.txt
cd game
python main.py --width 640 --fullscreen
```

The first start takes about ten seconds while the tracking models load. Two
players stand side by side about two metres from the camera; the game starts
once both are detected. Run it from inside `game/`, because the person
detector loads `yolov8n.pt` from the current folder.

Keys: `q` quit, `f` full screen, `s` hide the statistics panel, `d` hide the
webcam inset, `p` pause, `r` restart, `1`-`4` switch a task module off and on.
Useful options: `--width 960` for a sharper but slower picture, `--debug` to
draw the skeletons and boxes, `--source clip.mp4` to play from a video file.

All four tracking models are included in `models/` and `game/`, so nothing is
downloaded at run time.

## Tested environment

Windows 11, Intel Core i5-1235U, integrated graphics, no discrete GPU.

| Package | Version tested |
|---|---|
| Python | 3.13.9 |
| opencv-python | 5.0.0.93 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| mediapipe | 1.1.1 |
| ultralytics | 8.4.189 (installs PyTorch 2.14.1) |

The unit tests run with `python -m pytest game/tests`.

Task modules are loaded automatically from `game/modules/`. Any task whose
file does not exist yet is skipped, so the game always runs and the panel on
screen lists what is still missing. See `game/README.md` for how to add yours.

## Requirements

Each task owner writes the requirements for their own task in `docs/`.

- [Task 1, face](docs/requirements_task1.md)
- [Task 2, pose](docs/pose_module.md)
- [Task 3, identity](docs/requirements_task3.md)
- [Task 4, control](docs/requirements_task4.md)
- [Task 5, integration](docs/requirements_task5.md)
- [Task 5, measurement protocol](docs/task5_measurements.md)

## Layout

```
game/                the application
  main.py            the loop: camera, modules, timing, failure handling
  game_types.py      shared data types every module uses
  renderer.py        scene rendering and reactions to game events
  arena.py           the dojo backdrop the webcam picture sits in
  profiler.py        frame rate and latency measurement
  requirements.txt   python libraries needed
  modules/           one file per task
    face.py          task 1
    pose.py          task 2
    identity.py      task 3
    control.py       task 4
  tests/             task 4's unit tests
docs/                one requirements document per task
models/              the tracking models the modules load
```

Tasks 1 to 4 are each a module in `game/modules/`. Task 5 is not a module,
it is the application they plug into: `main.py`, `game_types.py`,
`renderer.py`, `arena.py` and `profiler.py`.
