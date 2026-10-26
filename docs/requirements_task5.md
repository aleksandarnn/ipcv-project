# Task 5 requirements: integration and scene

**Owner:** Aleksandar Nikolov (s2523477)
**Game:** Samurai Showdown, two players, one webcam

This is a first version with the basics. We add detail once the other tasks
exist and we know what actually causes trouble.

## What my task does

I own the program everything else plugs into. It opens the camera, runs the
four task modules every frame in the order identity, face, pose, control,
draws the result, and measures how long it all took.

I also draw the scene: the arena, the health bars, and the reactions when
something happens in the duel.

## What I will build

- One program that starts the whole game with a single command.
- Automatic loading of the four task modules, skipping any that do not exist yet.
- The arena background and the heads-up display with health and round state.
- Different visible reactions for a hit landing, a hit being blocked, and a miss.
- Frame rate and latency measurement, shown on screen and printed on exit.
- The game keeps running if one module crashes.

## Performance

Aiming for at least 20 frames per second with two players, and a delay of
under 100 ms between moving and seeing it on screen. The budgets now live in
`BUDGET_MS` in `game_types.py` and are the ones the task owners set
themselves: face 8 ms, pose 20 ms, identity 3 ms, control 3 ms, drawing 6 ms.

Nobody has measured a real frame yet. Once the modules run together we take
the numbers from `profiler.py` and change these to match reality.

## What I need from each of you

All the types are in `game/game_types.py`, and that file is the agreement,
not this document. Where the two disagree, the file wins.

**Task 1, faces.** A face position and size for each player, attached to the
right player. Leave it empty when you cannot find a face instead of guessing.

**Task 2, pose.** Keypoints with the names in `game_types.py`, each with an
honest confidence, smoothed inside the module. Coordinates are
crop-normalised against that player's `bbox`, and speed comes from `vx`,
`vy` and `speed` on the keypoint itself.

**Task 3, identity.** The same player keeps the same id across the whole
round, including when they cross, get blocked from view, or leave and come
back. The `bbox` is what pose and face crop against, so it has to be stable.

**Task 4, control.** Only the events listed in `game_types.py`, spelled
exactly. `damage` names the player who got hit, not the one who swung.

## Two rules for everyone

Nobody opens the camera except my main loop, and nobody draws on the frame.
Send me draw commands and I draw them in the right order.

## Not decided yet

How dodging is detected, since one webcam cannot see depth directly.

Whether attacks are resolved by reach or by timing.

Whether the game really needs legs. Ducking and jumping mean both players
must be fully in frame, which makes everyone smaller and makes faces and
keypoints harder. Worth deciding on purpose.

Task 4 is finished but sits on the `task4` branch and has not been merged.
Somebody needs to review it and pull it in.

The real budgets. The numbers in `BUDGET_MS` are still guesses; we replace
them with `profiler.py` output once all four modules run together.



