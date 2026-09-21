# TASK 5 - the main loop. Everything meets here.
#
# IPCV group project. Owner: Aleksandar Nikolov (s2523477).
#
# WHAT THIS FILE IS RESPONSIBLE FOR
#   - opening the camera (nobody else does)
#   - calling the four task modules in a fixed order, every frame
#   - timing each one, so we know who the bottleneck is
#   - catching module crashes so one bad module cannot kill the game
#   - rendering the scene and reacting to game events
#
# RUN IT
#   python main.py                      webcam, all four modules faked
#   python main.py --source clip.mp4    a video file instead of the camera
#   python main.py --real face          use the real face module, fake the rest
#   python main.py --width 640          smaller frames, higher frame rate
#
# KEYS
#   q  quit        s  hide/show the stats panel
#   r  restart     1  toggle the face module      2  toggle pose
#   p  pause       3  toggle identity             4  toggle control
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_types import BUDGET_MS, GameState, Module          # noqa: E402
from profiler import Profiler                               # noqa: E402
from renderer import Renderer                               # noqa: E402
from modules.stubs import (ControlStub, FaceStub,           # noqa: E402
                           IdentityStub, PoseStub)

# How many times a module may throw before we give up on it. The game keeps
# running without it and the stats panel says so, which is the "controlled
# behaviour during module failure" the brief asks for.
MAX_FAILURES = 5


def build_modules(real: list[str]) -> list[Module]:
    """One module per task, in the order they must run.

    Order matters. Faces come first because identity and pose both need them.
    Control runs last because it needs everything else to be up to date.
    """
    chosen: list[Module] = []
    for name, stub in (("face", FaceStub), ("pose", PoseStub),
                       ("identity", IdentityStub), ("control", ControlStub)):
        if name in real:
            chosen.append(load_real(name))
        else:
            chosen.append(stub())
    return chosen


def load_real(name: str) -> Module:
    """Import a teammate's real module.

    Each task owner writes modules/<name>.py containing a class called
    Module, matching the contract. Until that file exists this falls back to
    the stub, so a missing teammate never blocks anyone.
    """
    try:
        imported = __import__(f"modules.{name}", fromlist=["Module"])
        instance = imported.Module()
        print(f"[main] using the REAL {name} module")
        return instance
    except Exception as exc:                                  # noqa: BLE001
        print(f"[main] no real {name} module yet ({exc.__class__.__name__}), using the stub")
        from modules import stubs
        return {"face": stubs.FaceStub, "pose": stubs.PoseStub,
                "identity": stubs.IdentityStub, "control": stubs.ControlStub}[name]()


def open_source(source: str, width: int):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open the video source: {source}")
    if source.isdigit():
        # Ask the camera for a small frame. Asking is cheaper than resizing
        # afterwards, because the camera does it before we ever see the data.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 / 16))
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="IPCV project, main game loop")
    parser.add_argument("--source", default="0", help="webcam index or video file")
    parser.add_argument("--width", type=int, default=640, help="working frame width")
    parser.add_argument("--real", nargs="*", default=[],
                        help="modules to load for real: face pose identity control")
    parser.add_argument("--headless", type=int, default=0,
                        help="run N frames with no window, for testing")
    parser.add_argument("--snapshot", default="",
                        help="save one rendered frame to this path, for the report")
    args = parser.parse_args()

    cap = open_source(args.source, args.width)
    modules = build_modules(args.real)
    renderer = Renderer()
    profiler = Profiler()

    enabled = {m.name: True for m in modules}
    failures = {m.name: 0 for m in modules}
    show_stats, paused, frame_index = True, False, 0

    print("[main] running. q quits, s hides stats, r restarts, 1-4 toggle modules.")
    while True:
        profiler.start_frame()
        grabbed_at = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            if args.source.isdigit():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # loop a video file
            continue

        if frame.shape[1] != args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        # Mirror it, so moving left on screen means moving left in real life.
        frame = cv2.flip(frame, 1)

        state = GameState(frame_index=frame_index, timestamp=time.time())
        if paused:
            state.phase, state.message = "paused", "PAUSED"

        if not paused:
            for module in modules:
                if not enabled[module.name]:
                    continue
                started = time.perf_counter()
                try:
                    module.process(frame, state)
                except Exception as exc:                      # noqa: BLE001
                    failures[module.name] += 1
                    print(f"[main] {module.name} failed ({exc.__class__.__name__}: {exc})")
                    if failures[module.name] >= MAX_FAILURES:
                        enabled[module.name] = False
                        print(f"[main] {module.name} disabled after {MAX_FAILURES} failures")
                spent = time.perf_counter() - started
                profiler.record(module.name, spent)
                budget = BUDGET_MS.get(module.name)
                if budget and spent * 1000 > budget * 3:
                    print(f"[main] {module.name} took {spent*1000:.0f} ms, "
                          f"budget is {budget:.0f} ms")

        started = time.perf_counter()
        renderer.react(state)
        output = renderer.draw(frame, state)
        profiler.record("render", time.perf_counter() - started)

        offline = [name for name, on in enabled.items() if not on]
        if show_stats:
            renderer.stats_panel(output, profiler.lines(), offline)

        profiler.record_latency(time.perf_counter() - grabbed_at)
        profiler.end_frame()
        frame_index += 1

        if args.snapshot and frame_index == max(1, args.headless // 2):
            cv2.imwrite(args.snapshot, output)
            print(f"[main] snapshot written to {args.snapshot}")

        if args.headless:
            if frame_index >= args.headless:
                break
            continue

        cv2.imshow("IPCV project", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_stats = not show_stats
        elif key == ord("p"):
            paused = not paused
        elif key == ord("r"):
            for module in modules:
                module.reset()
            renderer.reset()
            frame_index = 0
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            name = ["face", "pose", "identity", "control"][key - ord("1")]
            enabled[name] = not enabled[name]
            print(f"[main] {name} {'on' if enabled[name] else 'off'}")

    cap.release()
    cv2.destroyAllWindows()
    print()
    print("PERFORMANCE SUMMARY")
    print(profiler.report())


if __name__ == "__main__":
    main()
