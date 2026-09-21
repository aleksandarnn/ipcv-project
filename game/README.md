# IPCV group project, game skeleton

Webcam-based multiplayer game. This is the frame that all five tasks plug into.

Maintained by **Aleksandar Nikolov (s2523477)**, task 5, integration.

## Run it

```
python main.py                    webcam, all four modules faked
python main.py --source clip.mp4  a video file instead of the camera
python main.py --real face        your real module, the rest still faked
python main.py --width 480        smaller frames, higher frame rate
```

Keys: `q` quit, `s` hide stats, `p` pause, `r` restart, `1`-`4` switch a
module on or off while it runs.

## What is here

```
game_types.py       the shared data types - READ THIS FIRST
main.py             the loop: camera, modules, timing, failures, rendering
profiler.py         per-module timing, frame rate, end-to-end latency
renderer.py         task 5: draws the scene and reacts to game events
modules/stubs.py    fake versions of all four other tasks
modules/<name>.py   your real module goes here
```

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

Then run `python main.py --real face`. If your file does not exist yet, or it
fails to import, the stub is used instead, so nobody is ever blocked waiting
for anybody else.

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

## Order the modules run in

`face` then `pose` then `identity` then `control`, then the renderer.

This is not arbitrary. Identity and pose both need faces to exist first, and
control needs everything else to be up to date before it decides anything.

## Why the stubs behave badly on purpose

The fake pose module jitters its keypoints and drops about 6% of them to low
confidence. The fake identity module counts how long a player has been
missing and fires a `player_lost` event.

That is deliberate. If the game only works with perfect input, it will fall
apart in the exhibition hall. Build against the bad input from day one.

## Performance

The loop prints a summary when it exits. That output goes straight into the
report, which asks for measured frame rate and latency:

```
average fps     : 351.0
average latency : 2.8 ms

per module, slowest first:
   render       1.50 ms
   pose         0.04 ms
   face         0.02 ms
```

Those numbers are with stubs, so they are a **baseline**, not a result. They
say the loop itself costs almost nothing, which means whatever we measure
later is the real modules. Re-measure after every integration.
