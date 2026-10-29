# Task 5 measurements: what to record in the team session

**Owner:** Aleksandar Nikolov (s2523477)

The report template asks my section for six things: end-to-end frame rate
with a low percentile, resource use with the laptop specification, long-run
stability, bootstrap time, the influence of lighting, and the fastest player
motion the system can still follow. Four of those need two people in a room,
so this is the list to work through once, in order, instead of discovering
afterwards that something was missed.

Everything is measured on one laptop so the numbers compare: mine, which goes
in the report as the test machine.

## Before anyone moves

Write down the laptop: CPU, RAM, whether a GPU is used, Windows version,
Python version, and the commit the test ran on (`git rev-parse --short HEAD`).
A frame rate without the machine it was measured on means nothing.

The program prints its own numbers when it exits, so the rule for the whole
session is: press `q` to quit, never close the window, and copy the
PERFORMANCE SUMMARY block into the log before starting the next run. Killing
the process loses the measurement.

## 1. Frame rate during normal gameplay

Two players, normal room light, both in frame, playing properly for about two
minutes. Then quit and copy the summary.

The summary gives mean fps, 5 % low, 1 % low, mean and worst latency, and the
per-module breakdown, with the first 30 frames left out as warm-up. The 5 %
low is the number the report actually wants alongside the mean, because it is
what a stutter feels like.

Run this twice at `--width 640` and twice at `--width 960`, so there is a
repetition and a resolution comparison rather than one lonely number.

## 2. Resource use

While run 1 is going, open Task Manager and note CPU percentage, memory in
MB, and GPU percentage if anything appears there. One reading in the middle
of the run is enough; it does not need to be a graph.

## 3. Long-run stability

One run left going for at least fifteen minutes with people moving in and out
of frame. Record how long it ran, whether it crashed, how many times the
picture froze, and the `stalls > 250 ms` count from the summary. If a module
goes OFFLINE in the stats panel, write down which one and after how long,
because that is the failure handling working rather than a bug.

## 4. Bootstrap time

The program now prints `playable after N seconds`, measured from launch to
the first frame on screen, which includes loading the four tracking models.
Do it three times and give the average, because the first launch after a
reboot is slower than the rest while Windows still has nothing cached.

Also record it once with a cold start, straight after a restart, since that
is the honest worst case.

## 5. Lighting

Three conditions, two players each, one minute each, nothing else changed:

- normal room light
- lights off, curtains open, so only daylight
- one bright lamp to one side, so half of each face is dark

For each one record the frame rate summary and, importantly, what broke
first. Expect the camera itself to slow down in the dark, because it holds
the shutter open longer, so the frame rate can fall even though the
processing has not changed. The `capture` line in the per-module list is
there to prove that, and it separates a camera problem from a code problem.

## 6. Fast motion

Players swing a hand slowly, then at normal speed, then as fast as they can,
ten swings each. Count how many of each group produced a reaction in the
game. The point of the test is to find the speed at which tracking starts
losing the limb, and the answer is partly fixed by the frame rate: at 25
frames a second a hand moving quickly travels a long way between frames, so
there is a limit no amount of tuning removes.

## Notes from the first solo measurements

Done alone, empty room, so pose and identity were cheap and these are not
gameplay numbers. They are here as a baseline and to record one thing that
mattered.

The camera backend turned out to be the bottleneck and not the vision code.
Through DirectShow the grab alone took 75 ms, holding the whole game to about
10 frames a second while the five modules together needed only 24 ms. Media
Foundation honours the same 16:9 sizes and delivers around 30, so the game
now asks for it first.

| Width | mean fps | 5 % low | mean latency | capture |
|---|---|---|---|---|
| 640, DirectShow | 9.9 | 7.5 | 101 ms | 75.4 ms |
| 640, Media Foundation | 29.9 | 14.7 | 33 ms | 7.9 ms |
| 960, Media Foundation | 26.5 | 15.7 | 38 ms | 5.9 ms |

With a person actually in frame pose costs about 23 ms rather than 11, which
is over its own 20 ms budget and makes it the slowest module. Valerii needs
to know that.
