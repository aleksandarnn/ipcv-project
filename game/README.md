# Samurai Showdown

A two-player webcam duel. Swing a hand to attack, cross both arms to block,
move out of the way to dodge. One webcam, two players, no controllers.

This is the frame that all five tasks plug into.

Maintained by **Aleksandar Nikolov (s2523477)**, task 5, integration.

## Run it

```
python main.py                    webcam
python main.py --source clip.mp4  a video file instead of the camera
python main.py --width 480        smaller frames, higher frame rate
```

Task modules are picked up automatically. Any of the four that does not exist
yet is simply skipped, and the panel on screen lists what is missing.

Keys: `q` quit, `s` hide stats, `p` pause, `r` restart, `1`-`4` switch a
module on or off while it runs.

## What is here

```
game_types.py       the shared data types - READ THIS FIRST
main.py             the loop: camera, modules, timing, failures, rendering
profiler.py         per-module timing, frame rate, end-to-end latency
arena.py            task 5: the dojo backdrop the webcam picture sits in
renderer.py         task 5: draws the scene and reacts to game events
modules/identity.py task 3, runs first  (Akash)
modules/face.py     task 1
modules/pose.py     task 2             (Valerii)
modules/control.py  task 4             (Vaishnav, on the task4 branch)
```

The pose and identity modules need `mediapipe` and `ultralytics`, plus the
model files in `models/`. See `requirements.txt`.

## How to add your module

Create `modules/face.py`, `modules/pose.py`, `modules/identity.py` or
`modules/control.py` depending on your task. Put a class called `Module` in
it, subclassing `Module` from `game_types.py`:

```python
from game_types import Module as Base

class Module(Base):
    name = "face"

    def process(self, frame, state):
        # read the frame, write your results into state
        ...
```

Then just run `python main.py`. Your module is found automatically. Tasks
whose file does not exist yet are skipped, so nobody is blocked waiting for
anybody else.

## The four rules

1. **Only `main.py` opens the camera.** Your module receives a frame.
2. **Never draw on the frame.** Append `DrawCommand`s to `state.draw` and the
   renderer draws them in the right order. Otherwise two people fight over
   the same pixels.
3. **Respect your time budget.** See `BUDGET_MS` in `game_types.py`. At 25 fps
   the whole frame has 40 ms. If you need more, say so and we renegotiate,
   but somebody else then gets less.
4. **Do not crash the game.** `main.py` catches your exceptions, but after
   five failures your module is switched off and the stats panel says
   `OFFLINE` for everyone to see.

## The events

Task 4 raises these, task 5 makes them visible. The full list lives in
`game_types.py`. Spelling matters: an event nobody recognises is ignored
silently, so agree any new one first.

| event | when | what the scene does |
|---|---|---|
| `attack` | a player swings | trail on the swinging arm |
| `block` | both arms crossed | guard glow |
| `dodge` | moved clear in time | brief motion streak |
| `damage` | a hit landed | hard shake, red wash |
| `blocked` | a guard stopped it | white spark, almost no shake |
| `missed` | the swing hit air | barely anything |
| `ko` | health reached zero | full shake and flash |

The three outcomes deliberately look completely different, so players can
tell instantly whether a swing landed, was blocked, or missed.

## Order the modules run in

`identity` then `face` then `pose` then `control`, then the renderer.
This is what `TASKS` in `main.py` actually does.

Identity runs first because it finds the people and gives each one a box and
a stable id. Face and pose both work inside that box. Control runs last,
because it needs everything else to be up to date before it decides anything.

## Units — read this before you draw anything

Two things in `game_types.py` are easy to get wrong:

- `player.bbox` is a `BBox` in **full-frame pixels**, with `.x0 .y0 .x1 .y1`
  and `.w .h`.
- `player.keypoints` are **crop-normalised**: x and y run 0 to 1 across that
  player's box, not across the screen.

So to put something on a wrist, convert first:

```python
kp = player.point("right_wrist")
if kp and player.bbox:
    x = player.bbox.x0 + kp.x * player.bbox.w
    y = player.bbox.y0 + kp.y * player.bbox.h
```

Use the keypoint values straight as pixels and everything you draw lands in
the top-left corner.

Speed comes from the keypoint itself: `kp.vx`, `kp.vy` and `kp.speed`, in
crop-widths per second. There is no longer a separate `motion` dictionary.

## Performance

The loop prints a summary when it exits. That output goes straight into the
report, which asks for measured frame rate and latency:

```
average fps     : 558.9
average latency : 1.8 ms

per module, slowest first:
   render       0.14 ms
```


