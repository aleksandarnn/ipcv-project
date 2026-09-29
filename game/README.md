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
renderer.py         task 5: draws the scene and reacts to game events
modules/face.py     task 1, written by its owner
modules/pose.py     task 2
modules/identity.py task 3
modules/control.py  task 4
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

`face` then `pose` then `identity` then `control`, then the renderer.

This is not arbitrary. Identity and pose both need faces to exist first, and
control needs everything else to be up to date before it decides anything.

## Performance

The loop prints a summary when it exits. That output goes straight into the
report, which asks for measured frame rate and latency:

```
average fps     : 558.9
average latency : 1.8 ms

per module, slowest first:
   render       0.14 ms
```


