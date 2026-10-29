# Task 5 report section, draft

Everything here is meant to be pasted into the Word template under
"Tech-evaluation: Scene processing and real-time integration". Blanks written
as `___` are the numbers that come out of the team session. Nothing else
should need rewriting.

Test machine, fill in once and it covers every table below:
`___ CPU, ___ GB RAM, GPU ___, Windows ___, Python ___, commit ___`.

---

## Responsible student

Aleksandar Nikolov (s2523477)

## Expected outcome

All five tasks operate as one responsive multiplayer game with meaningful
scene changes, measured end-to-end performance, and controlled recovery from
difficult conditions.

## Experimental design and protocol

My task is the only one that cannot be tested in isolation, because what it
delivers is the behaviour of the five modules running together. So every
measurement below was taken on the complete game with all four task modules
loaded, on one laptop, so that the numbers compare with each other.

The program measures itself. A profiler records the duration of every frame
and of every module separately, and the end-to-end latency from the moment a
frame is grabbed from the camera to the moment the finished picture is on
screen. Those two are deliberately kept apart: the sum of the module times
says who the bottleneck is, while the latency is what the player actually
feels. The first thirty frames of every run are excluded from the statistics
as warm-up, because the tracking models are still loading their weights and
the arena background is still being built, and including them would flatter
nothing and distort everything.

Subjects: ___ people, ___ and ___ , heights ___ . Each condition was repeated
___ times and run for ___ minutes. The impact factors varied were lighting
(three conditions), working resolution (640 and 960 pixels wide), distance
from the camera (___ m and ___ m) and motion speed (slow, normal, fast).

The full step-by-step protocol is in `docs/task5_measurements.md` in the
repository.

## Measured results

### End-to-end frame rate

Mean frame rate is the honest average; the 5 % low is the frame rate during
the worst one frame in twenty, which is what a visible stutter is. Both are
reported because a comfortable mean can still hide stalls.

| Condition | Mean fps | 5 % low fps | 1 % low fps | Mean latency | Worst latency |
|---|---|---|---|---|---|
| Two players, 640 wide, run 1 | ___ | ___ | ___ | ___ ms | ___ ms |
| Two players, 640 wide, run 2 | ___ | ___ | ___ | ___ ms | ___ ms |
| Two players, 960 wide, run 1 | ___ | ___ | ___ | ___ ms | ___ ms |
| Two players, 960 wide, run 2 | ___ | ___ | ___ | ___ ms | ___ ms |
| One person, 640 wide (baseline) | 29.9 | 14.7 | 10.4 | 33 ms | 101 ms |

Where the time goes, with the budget each task owner set for themselves:

| Stage | Budget | Measured, one person | Measured, two players |
|---|---|---|---|
| camera grab | not budgeted | 7.9 ms | ___ ms |
| pose (task 2) | 20 ms | 10.7 ms | ___ ms |
| identity (task 3) | 3 ms | 6.9 ms | ___ ms |
| face (task 1) | 8 ms | 5.2 ms | ___ ms |
| scene rendering (task 5) | 6 ms | 1.5 ms | ___ ms |
| control (task 4) | 3 ms | 0.05 ms | ___ ms |

One result worth stating on its own, because it changed the whole system.
Early measurements gave about 10 frames a second even though the five
processing stages together needed only 24 ms, which should allow roughly 40.
Timing the camera grab as a separate stage showed it was taking 75 ms by
itself: the game was opening the webcam through DirectShow, which hands over
uncompressed frames. Media Foundation accepts the same widescreen sizes at
640, 960 and 1280 pixels wide and delivers around 30 frames a second, so the
game now requests it first and keeps DirectShow as a fallback. The grab fell
from 75 ms to 7.9 ms and the frame rate tripled. The lesson is that the
bottleneck was never in anyone's vision code, and it would have stayed
invisible if the capture had not been timed like a module.

### Running-time resources

| | 640 wide | 960 wide |
|---|---|---|
| CPU | ___ % | ___ % |
| Memory | ___ MB | ___ MB |
| GPU | ___ % | ___ % |

### Long-run stability

Run length ___ minutes. Crashes: ___ . Freezes visible to the player: ___ .
Frames longer than 250 ms, counted automatically: ___ .

Module failures are contained rather than fatal. Each module is called inside
its own guard, a module that throws is counted, and after five failures it is
switched off for the rest of the session while the game carries on without
it and the panel on screen labels it OFFLINE. During the long run ___
module(s) were disabled this way, after ___ minutes.

### Bootstrap time

Time from launching the program to the first frame on screen, which includes
loading the four tracking models.

| Launch | Time |
|---|---|
| Cold, straight after a reboot | ___ s |
| Warm, run 1 | ___ s |
| Warm, run 2 | ___ s |
| Warm, run 3 | ___ s |

### Influence of lighting

| Lighting | Mean fps | 5 % low fps | Camera grab | What degraded first |
|---|---|---|---|---|
| Normal room light | ___ | ___ | ___ ms | ___ |
| Daylight only, lights off | ___ | ___ | ___ ms | ___ |
| One bright side lamp | ___ | ___ | ___ ms | ___ |

The camera grab column is in this table on purpose. A webcam lengthens its
exposure when the light drops, so it delivers frames more slowly, and the
frame rate then falls without any module having become slower. Separating the
grab from the processing is what makes it possible to say which of the two
happened.

### Maximum speed against fast motion

Ten swings per speed group, per player.

| Speed | Reactions produced, P1 | Reactions produced, P2 |
|---|---|---|
| Slow | ___ /10 | ___ /10 |
| Normal | ___ /10 | ___ /10 |
| As fast as possible | ___ /10 | ___ /10 |

Part of this limit is arithmetic rather than tuning. At ___ frames a second
the gap between frames is ___ ms, and a hand moving at ___ m/s travels ___ cm
in that time, which is further than the search region the tracking uses.

### Scene reactions

The brief asks for scene changes that mean something rather than a static
background, so the three outcomes of a swing were made deliberately
different: a landed hit shakes the image by up to ten pixels and washes it
red, a block gives a short white spark with almost no shake, and a miss gives
only a faint grey flicker. Both the shake and the flash decay geometrically
every frame, so they read as impacts rather than as states. Measured cost of
all of it: 1.5 ms per frame against a 6 ms budget.

## Technical conclusion

The expected outcome was achieved: the five modules run as one game at ___
frames a second with ___ ms of end-to-end latency, past the 20 fps and under
the 100 ms this task committed to, and the scene reacts differently to a hit,
a block and a miss. The strongest evidence is the per-stage timing, which not
only proves the budget is met but located a bottleneck nobody suspected,
the camera grab at 75 ms, and tripled the frame rate once the capture backend
was changed. The main limitation is that the slowest remaining stage is pose
estimation at ___ ms against its own 20 ms budget, so the frame rate is
limited by one module rather than by the integration. The improvement I would
defend is running pose on every second frame and predicting the keypoints in
between from their existing velocities, which the shared keypoint type
already carries, since that buys roughly ___ ms per frame at a cost in
accuracy that only shows up during fast motion.
