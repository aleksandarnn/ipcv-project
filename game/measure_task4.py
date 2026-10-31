# TASK 4 - measuring the metrics the evaluation report asks for.
#
# IPCV group project. Owner: Vaishnav Rengan (task 4).
#
# WHAT THIS FILE DOES
#   Runs the normal live game (identity, face, pose, control, renderer) and
#   tells two players on screen who must do which action and when. It records
#   what the control module recognised and prints the Task 4 metrics with the
#   same names as the report template. It does not change any game code: it
#   only reads the events control raises and, for details an event does not
#   carry (swing direction, duck or step), control's own state.
#
# RUN IT (both players in view BEFORE starting; P1 / P2 are shown above the
# bodies, and the cue says where the acting player stands)
#   python measure_task4.py --condition room          normal room light, 50 tries
#   python measure_task4.py --condition room --reps 2 --both-reps 0   20 tries
#   python measure_task4.py --condition dim           same test, other light
#   python measure_task4.py --compare                 lighting table of all runs
#
# KEYS
#   x  redo this trial (the player did the wrong thing)
#   p  pause            q  stop early and save what was measured
#
# THE PROTOCOL (50 tries with the defaults)
#   1. Single actions: for each of the 5 actions, P1 does it --reps times (4),
#      then P2. The other player stands still. 5 x 2 x 4 = 40 tries.
#   2. Both players together: both do the same action at the same moment,
#      --both-reps times per action (2). 5 x 2 = 10 tries.
#   3. Idle: both players stand, talk and move a little, no actions (--idle s).
#   Every trial is: GET READY (1.5 s), GO (2 s to act), relax (1 s). A trial
#   only counts if the round was running during GO; a K.O. or a lost player
#   then makes the script repeat it.
#
# THE METRICS (template names)
#   Action recognition precision  per action: accuracy = cued trials recognised
#                                 as that action; precision = of everything
#                                 recognised as that action, the share that was
#                                 really cued
#   False triggers possibility    actions recognised when nobody was cued, per
#                                 minute, and what they were
#   Missed action                 % cued actions with no recognition at all,
#                                 and % recognised as the wrong action
#   Command/signal latency        from the movement onset (seen by pose) to the
#                                 frame with the response on screen
#   Simultaneous two players      % "both" trials where both were right
#   Influence of lighting         run once per --condition, then --compare
#
# WHY A TRY FAILED
#   A try only starts once pose sees both bodies (control has their shoulders),
#   so a missing skeleton is not counted as a Task 4 miss by accident. During GO
#   the script records the values the rules compare with their thresholds (hand
#   speed, wrist confidence, shoulder drop, sideways shift, and any freeze,
#   grace period or edge block), names the reason for every missed try, and
#   writes all values per frame to task4_<condition>_frames.csv. The camera
#   picture is saved as task4_<condition>.avi, so a session can be replayed
#   with --source. Tune on a replay, but report numbers from a fresh session.
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_types import GameState, player_label    # noqa: E402
from main import load_modules, open_source        # noqa: E402
from modules import control as rules              # noqa: E402  thresholds, _velocity
from renderer import Renderer                     # noqa: E402

RESULTS = Path(__file__).resolve().parent / "measurements"
WINDOW = "Task 4 measurement"

ACTIONS = ("hcut", "vcut", "guard", "duck", "step")
NAMES = {"hcut": "HORIZONTAL CUT", "vcut": "VERTICAL CUT", "guard": "GUARD (cross arms)",
         "duck": "DUCK", "step": "SIDESTEP"}
READY_S, GO_S, REST_S = 1.5, 2.0, 1.0

# Movement onset, from pose. Below these speeds (shoulder widths per second)
# the body counts as still; the strike starts where it goes above them.
ONSET_ARM = 1.5          # a wrist relative to the shoulders: cuts and guard
ONSET_BODY = 0.5         # the shoulder midpoint: duck (down) and step (sideways)
ONSET_EARLY_S = 0.3      # players often start a little before GO
SKELETON_S = 0.5         # pose must have given shoulders this recently to start a try
FIELDS = ("seen", "missing", "frozen", "grace", "edge", "guard",
          "speed", "reach", "outward", "conf", "drop", "shift")


# ================================================================ diagnosis
def rule_inputs(ctl, pid: int, player, t: float) -> dict:
    """The values control's rules compare with their thresholds, this frame."""
    row = dict.fromkeys(FIELDS, 0)
    row["missing"] = player.frames_missing if player is not None else -1
    f = ctl.fighters.get(pid)
    if f is None:
        return row
    row.update(seen=int(f.last_t == t), frozen=int(f.frozen), edge=int(f.at_edge),
               grace=int(not ctl._active(f, t)), guard=int(f.guard))
    s = f.scale
    if not row["seen"] or not s:
        return row
    for name, hist in f.wrists.items():
        v = rules._velocity(hist)
        if v is None:
            continue
        speed = math.hypot(*v) / s
        if speed > row["speed"]:
            _, rx, ry = hist[-1]
            kp = player.keypoints.get(name) if player is not None else None
            row.update(speed=speed, reach=math.hypot(rx, ry) / s,
                       outward=int(rx * v[0] + ry * v[1] > 0),
                       conf=kp.confidence if kp is not None else 0.0)
    if f.baseline_y is not None:
        row["drop"] = (f.ty - f.baseline_y) / s
    row["shift"] = max((abs(f.tx - x0) for t0, x0, _ in f.torso
                        if t - t0 <= rules.STEP_WINDOW_S), default=0.0) / s
    return row


def new_diag() -> dict:
    return {"go_frames": 0, "body_frames": 0, "max_speed": 0.0, "fast_conf": 0.0,
            "candidate": False, "max_drop": 0.0, "max_shift": 0.0, "guard": False,
            "blocked": {}}


def track(diag: dict, row: dict) -> None:
    """Keep the peak values of one acting player during GO."""
    diag["go_frames"] += 1
    if not row["seen"]:
        return
    diag["body_frames"] += 1
    diag["max_speed"] = max(diag["max_speed"], row["speed"])
    diag["max_drop"] = max(diag["max_drop"], row["drop"])
    diag["max_shift"] = max(diag["max_shift"], row["shift"])
    diag["guard"] = diag["guard"] or bool(row["guard"])
    if row["speed"] >= rules.ATTACK_SPEED:
        diag["fast_conf"] = max(diag["fast_conf"], row["conf"])
        if (row["conf"] >= rules.ATTACK_MIN_CONF and row["reach"] >= rules.MIN_REACH
                and row["outward"]):
            diag["candidate"] = True       # every attack condition met this frame
            for block in ("grace", "edge", "guard", "frozen"):
                if row[block]:
                    diag["blocked"][block] = diag["blocked"].get(block, 0) + 1


def reason(trial: dict, pid: str) -> str:
    """Why this player's cued action was (not) recognised."""
    action, first = trial["action"], trial["first"].get(pid)
    if first == action:
        return "recognised"
    d = trial["diag"][pid]
    if not d["go_frames"] or d["body_frames"] < 0.5 * d["go_frames"]:
        return "no body points from pose"
    if first is not None:
        return "recognised as another action"
    if action in ("hcut", "vcut"):
        if d["max_speed"] < rules.ATTACK_SPEED:
            return "hand too slow"
        if d["fast_conf"] < rules.ATTACK_MIN_CONF:
            return "wrist confidence too low"
        if not d["candidate"]:
            return "hand too close or not moving outward"
        return "blocked (grace, edge, guard or cooldown)"
    if action == "duck":
        return "shoulders did not drop enough" if d["max_drop"] < rules.DUCK_DROP \
            else "blocked or other"
    if action == "step":
        return "body did not shift enough" if d["max_shift"] < rules.STEP_SHIFT \
            else "blocked or other"
    return "crossed wrists not seen"


# ================================================================ recognition
def recognitions(state, ctl, guard_before: set) -> list[tuple[int, str]]:
    """What control recognised this frame, as (player_id, action)."""
    found = []
    slashes = [fx for fx in ctl.fx if fx.kind == "slash" and fx.t0 == state.timestamp]
    for e in state.events:
        fighter = ctl.fighters.get(e.player_id)
        if e.kind == "attack" and fighter is not None:
            # The slash effect ends where the blade was heading, so it holds the
            # swing direction with the same rule control used (|vx| >= |vy|).
            mine = min(slashes, default=None,
                       key=lambda fx: math.hypot(fx.x - fighter.tx, fx.y - fighter.ty))
            if mine is not None and mine.points:
                dx, dy = mine.points[-1][0] - mine.x, mine.points[-1][1] - mine.y
                found.append((e.player_id, "hcut" if abs(dx) >= abs(dy) else "vcut"))
            else:
                found.append((e.player_id, "cut?"))
        elif e.kind == "dodge" and fighter is not None:
            found.append((e.player_id, "duck" if fighter.ducking else "step"))
        elif e.kind == "block" and e.player_id not in guard_before:
            # block is sent every frame while the guard is held; count it once
            found.append((e.player_id, "guard"))
    guard_before.clear()
    guard_before.update(e.player_id for e in state.events if e.kind == "block")
    return found


def body_sample(player, ctl):
    """Shoulder midpoint, wrists relative to it, and the body scale, in pixels."""
    crop = player.crop
    ls, rs = player.point("left_shoulder"), player.point("right_shoulder")
    if crop is None or ls is None or rs is None:
        return None

    def px(kp):
        return crop.x0 + kp.x * crop.w, crop.y0 + kp.y * crop.h

    (lx, ly), (rx, ry) = px(ls), px(rs)
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    fighter = ctl.fighters.get(player.player_id)
    s = (fighter.scale if fighter is not None and fighter.scale else 0) \
        or max(1.0, math.hypot(lx - rx, ly - ry))
    wrists = {}
    for name in ("left_wrist", "right_wrist"):
        kp = player.point(name)
        if kp is not None:
            wx, wy = px(kp)
            wrists[name] = (wx - mx, wy - my)
    return mx, my, s, wrists


def onset(history, action: str, since: float, until: float) -> float | None:
    """Capture time of the frame where the recognised movement began.

    That is the start of the last continuous movement before the recognition,
    so lifting the arm for a cut is not counted, only the strike itself: a
    pause or a turn of the wrist (up, then down) starts a new movement.
    """
    frames = [h for h in history if since <= h[0] <= until]
    start = began = last_v = None
    for (t0, x0, y0, _, w0), (t1, x1, y1, s, w1) in zip(frames, frames[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        if action in ("hcut", "vcut", "guard"):
            moves = [((w1[n][0] - w0[n][0]) / dt / s, (w1[n][1] - w0[n][1]) / dt / s)
                     for n in w1 if n in w0]
            v = max(moves, default=(0.0, 0.0), key=lambda m: math.hypot(*m))
            moving = math.hypot(*v) > ONSET_ARM
            if moving and last_v is not None and v[0] * last_v[0] + v[1] * last_v[1] < 0:
                start = None                   # the wrist turned: a new movement
            last_v = v
        elif action == "duck":
            moving = (y1 - y0) / dt / s > ONSET_BODY
        else:
            moving = abs(x1 - x0) / dt / s > ONSET_BODY
        if moving:
            start = t0 if start is None else start
            began = start
        else:
            start = None
    return began


# ================================================================== protocol
def make_trials(reps: int, both_reps: int) -> list[dict]:
    trials = []
    for action in ACTIONS:
        for pid in (0, 1):
            trials += [{"kind": "single", "action": action, "actors": [pid]}
                       for _ in range(reps)]
    for action in ACTIONS:
        trials += [{"kind": "both", "action": action, "actors": [0, 1]}
                   for _ in range(both_reps)]
    return trials


def redo(trial: dict) -> dict:
    """The same trial again, with nothing recorded."""
    return {k: trial[k] for k in ("kind", "action", "actors")}


def finish(trial: dict, data: dict) -> None:
    """A completed trial: its recognitions and false events now count."""
    trial["done"] = True
    data["recognised"] += trial.pop("recognised")
    for event in trial.pop("false", []):
        event["trial"] = len(data["trials"])     # so a run can be checked afterwards
        data["false"].append(event)
    data["trials"].append(trial)


def side(ctl, pid: int) -> str:
    """Where this player stands right now, from control's own body position.

    Identity numbers the players, and that number only matches left/right if
    both stood in view when the game started, so never assume it.
    """
    me, other = ctl.fighters.get(pid), ctl.fighters.get(1 - pid)
    if me is None or other is None:
        return "?"
    return "LEFT" if me.tx < other.tx else "RIGHT"


def name_tags(img, ctl) -> None:
    """P1 / P2 above each body, so the players know which one they are."""
    for pid, f in ctl.fighters.items():
        if f.frozen or not f.scale:
            continue
        text = player_label(pid)
        x, y = int(f.tx), int(max(30, f.ty - 2.3 * f.scale))
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(img, text, (x - size[0] // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(img, text, (x - size[0] // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3, cv2.LINE_AA)


def banner(img, lines: list[tuple[str, tuple]]) -> None:
    """The cue box at the bottom of the picture. Drawn by this script, not a module."""
    h, w = img.shape[:2]
    box = int(h * 0.2)
    shade = img[h - box:, :].copy()
    cv2.rectangle(shade, (0, 0), (w, box), (0, 0, 0), -1)
    img[h - box:, :] = cv2.addWeighted(shade, 0.7, img[h - box:, :], 0.3, 0)
    y = h - box + int(box * 0.38)
    for i, (text, colour) in enumerate(lines):
        scale = (1.3 if i == 0 else 0.7) * w / 1280
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(img, text, ((w - size[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 3 if i == 0 else 2, cv2.LINE_AA)
        y += int(box * 0.32)


def who(trial, ctl) -> str:
    if trial["kind"] == "both":
        return "BOTH PLAYERS"
    pid = trial["actors"][0]
    return f"{player_label(pid)} (the person on the {side(ctl, pid)})"


# =================================================================== metrics
def summarise(data: dict) -> dict:
    trials = [t for t in data["trials"] if t.get("done")]
    recognised = Counter(data["recognised"])
    out = {"condition": data["condition"], "actions": {}, "mean_fps": data["mean_fps"]}

    # every correct first recognition, from single and from "both" trials
    right = Counter(t["action"] for t in trials for pid in t["actors"]
                    if t["first"].get(str(pid)) == t["action"])
    latencies = []
    for action in ACTIONS:
        cued = [(t, pid) for t in trials if t["kind"] == "single" and t["action"] == action
                for pid in t["actors"]]
        firsts = [t["first"].get(str(pid)) for t, pid in cued]
        correct = sum(f == action for f in firsts)
        missed = sum(f is None for f in firsts)
        lat = [t["latency_ms"][str(pid)] for t, pid in cued
               if str(pid) in t.get("latency_ms", {})]
        latencies += lat
        n = len(cued)
        out["actions"][action] = {
            "cued": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "precision": right[action] / recognised[action] if recognised[action] else None,
            "missed": missed / n if n else None,
            "wrong": (n - correct - missed) / n if n else None,
            "confused_with": dict(Counter(f for f in firsts if f not in (None, action))),
            "latency_ms": statistics.median(lat) if lat else None,
        }

    both = [t for t in trials if t["kind"] == "both"]
    ok = [t for t in both if all(t["first"].get(str(p)) == t["action"] for p in t["actors"])]
    out["simultaneous"] = {
        "trials": len(both),
        "success": len(ok) / len(both) if both else None,
        "per_action": {a: f"{sum(t in ok for t in both if t['action'] == a)}/"
                          f"{sum(t['action'] == a for t in both)}" for a in ACTIONS},
    }
    for t in both:
        latencies += list(t.get("latency_ms", {}).values())

    false = data["false"]
    idle_false = [f for f in false if f["stage"] == "idle"]
    out["false_triggers"] = {
        "idle_per_min": len(idle_false) / (data["idle_s"] / 60) if data["idle_s"] else None,
        "idle_count": len(idle_false),
        "idle_minutes": data["idle_s"] / 60,
        "all_per_min": len(false) / (data["play_s"] / 60) if data["play_s"] else None,
        "all_count": len(false),
        "by_action": dict(Counter(f["action"] for f in false)),
        "by_stage": dict(Counter(f["stage"] for f in false)),
    }
    out["latency_ms"] = {
        "median": statistics.median(latencies) if latencies else None,
        "mean": statistics.mean(latencies) if latencies else None,
        "p90": sorted(latencies)[int(0.9 * (len(latencies) - 1))] if latencies else None,
        "n": len(latencies),
    }

    # Why tries failed, and Task 4 on its own: only tries where pose saw the body.
    single = [(t, str(pid)) for t in trials if t["kind"] == "single" and "diag" in t
              for pid in t["actors"]]
    reasons = [(t, p, reason(t, p)) for t, p in single]
    out["reasons"] = dict(Counter(r for _, _, r in reasons).most_common())
    seen = [(t, p) for t, p, r in reasons if r != "no body points from pose"]
    out["task4_only"] = {"tries": len(seen),
                         "correct": sum(t["first"].get(p) == t["action"] for t, p in seen)}

    def median_peak(actions, key):
        vals = [t["diag"][p][key] for t, p in seen if t["action"] in actions]
        vals = [v for v in vals if v]
        return statistics.median(vals) if vals else None

    out["peaks"] = {
        "cut hand speed": (median_peak(("hcut", "vcut"), "max_speed"), rules.ATTACK_SPEED),
        "cut wrist confidence when fast": (median_peak(("hcut", "vcut"), "fast_conf"),
                                           rules.ATTACK_MIN_CONF),
        "duck shoulder drop": (median_peak(("duck",), "max_drop"), rules.DUCK_DROP),
        "sidestep shift": (median_peak(("step",), "max_shift"), rules.STEP_SHIFT),
    }
    out["body_seen"] = {p: f"{sum(t['diag'][p]['body_frames'] for t, q in single if q == p)}"
                           f"/{sum(t['diag'][p]['go_frames'] for t, q in single if q == p)}"
                        for p in ("0", "1")}
    return out


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f} %"


def ms(v) -> str:
    return "-" if v is None else f"{v:.0f} ms"


def report(s: dict) -> str:
    rows = [f"TASK 4 METRICS - condition: {s['condition']}   (mean fps {s['mean_fps']:.1f})",
            "",
            "Action recognition precision / Missed action and recognition",
            f"  {'action':<16}{'cued':>5}{'accuracy':>10}{'precision':>11}"
            f"{'missed':>8}{'wrong':>7}{'latency':>9}  confused with"]
    for a in ACTIONS:
        r = s["actions"][a]
        rows.append(f"  {NAMES[a].split(' (')[0].lower():<16}{r['cued']:>5}"
                    f"{pct(r['accuracy']):>10}{pct(r['precision']):>11}{pct(r['missed']):>8}"
                    f"{pct(r['wrong']):>7}{ms(r['latency_ms']):>9}  "
                    f"{r['confused_with'] or '-'}")
    f = s["false_triggers"]

    def per_min(v):
        return "-" if v is None else f"{v:.1f}"

    rows += ["",
             "False triggers possibility",
             f"  idle: {f['idle_count']} in {f['idle_minutes']:.1f} min = "
             f"{per_min(f['idle_per_min'])} per minute",
             f"  whole session, outside cues: {f['all_count']} = "
             f"{per_min(f['all_per_min'])} per minute   "
             f"by action {f['by_action'] or '-'}   when {f['by_stage'] or '-'}",
             "",
             "Command/signal latency (movement onset to response on screen)",
             f"  median {ms(s['latency_ms']['median'])}, mean {ms(s['latency_ms']['mean'])}, "
             f"90 % within {ms(s['latency_ms']['p90'])}  (n = {s['latency_ms']['n']})",
             "",
             "Simultaneous-two-players action recognized success rate",
             f"  {pct(s['simultaneous']['success'])} of {s['simultaneous']['trials']} trials"
             f"   per action {s['simultaneous']['per_action']}"]
    if s["reasons"]:
        only = s["task4_only"]
        rows += ["",
                 "Why single tries were (not) recognised",
                 *(f"  {r:<42}{n:>3}" for r, n in s["reasons"].items()),
                 f"  Task 4 alone (tries where pose saw the body): {only['correct']} of "
                 f"{only['tries']} correct",
                 "  frames with body points during GO: " + ", ".join(
                     f"{player_label(int(p))} {v}" for p, v in s["body_seen"].items()),
                 "",
                 "Peak values during GO, median over tries (rule threshold)"]
        for name, (value, limit) in s["peaks"].items():
            rows.append(f"  {name:<34}{'-' if value is None else f'{value:.2f}':>6}"
                        f"   ({limit})")
    return "\n".join(rows)


def compare() -> None:
    """Influence of lighting conditions: one line per measured condition."""
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob("task4_*.json"))]
    if not runs:
        raise SystemExit(f"No runs in {RESULTS}. Measure first with --condition.")
    print("Influence of lighting conditions")
    print(f"  {'condition':<12}{'fps':>6}{'accuracy':>10}{'missed':>8}"
          f"{'false/min':>11}{'latency':>9}{'both ok':>9}")
    for data in runs:
        s = summarise(data)
        acc = [r["accuracy"] for r in s["actions"].values() if r["accuracy"] is not None]
        mis = [r["missed"] for r in s["actions"].values() if r["missed"] is not None]
        fpm = s["false_triggers"]["idle_per_min"]
        print(f"  {s['condition']:<12}{s['mean_fps']:>6.1f}"
              f"{pct(sum(acc) / len(acc) if acc else None):>10}"
              f"{pct(sum(mis) / len(mis) if mis else None):>8}"
              f"{'-' if fpm is None else f'{fpm:.1f}':>11}"
              f"{ms(s['latency_ms']['median']):>9}{pct(s['simultaneous']['success']):>9}")


# ===================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 live measurement")
    parser.add_argument("--condition", default="room",
                        help="name of the lighting/distance condition, e.g. room, dim")
    parser.add_argument("--reps", type=int, default=4,
                        help="tries per action per player, alone")
    parser.add_argument("--both-reps", type=int, default=2,
                        help="tries per action with both players together")
    parser.add_argument("--idle", type=float, default=60.0, help="seconds of idle test")
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--headless", type=int, default=0,
                        help="run N frames with no window, for testing the script")
    parser.add_argument("--no-video", action="store_true",
                        help="do not save the camera picture")
    parser.add_argument("--compare", action="store_true",
                        help="print the lighting comparison of all saved runs")
    args = parser.parse_args()
    if args.compare:
        compare()
        return

    RESULTS.mkdir(exist_ok=True)
    frames_file = open(RESULTS / f"task4_{args.condition}_frames.csv", "w", newline="",
                       encoding="utf-8")
    frames_log = csv.writer(frames_file)
    frames_log.writerow(["t", "frame", "trial", "stage", "cued", "actors"]
                        + [f"{player_label(p)}_{k}" for p in (0, 1) for k in FIELDS])
    video = None
    cap = open_source(args.source, args.width)
    modules, _ = load_modules()
    ctl = next((m for m in modules if m.name == "control"), None)
    if ctl is None:
        raise SystemExit("modules/control.py did not load, nothing to measure")
    for m in modules:
        if m.name == "pose" and any(o.name == "identity" for o in modules):
            m.allow_stubs = False
    renderer = Renderer()

    queue = make_trials(args.reps, args.both_reps)
    total = len(queue)
    data = {"condition": args.condition, "reps": args.reps, "both_reps": args.both_reps,
            "trials": [], "recognised": [], "false": [], "idle_s": 0.0, "play_s": 0.0,
            "mean_fps": 0.0}
    history = {0: deque(maxlen=60), 1: deque(maxlen=60)}
    guard_before: set = set()
    trial, stage, stage_t = None, "ready", 0.0
    idle_left = args.idle
    paused, frame_index, started, last_t = False, 0, time.time(), None
    print(f"[measure] {total} tries, then {args.idle:.0f} s idle")

    while True:
        ok, frame = cap.read()
        captured = time.time()
        if not ok:
            break
        if not args.no_video and not args.headless:
            if video is None:
                # the raw camera picture, so --source can replay it exactly
                video = cv2.VideoWriter(str(RESULTS / f"task4_{args.condition}.avi"),
                                        cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                        (frame.shape[1], frame.shape[0]))
            video.write(frame)
        if frame.shape[1] != args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)

        state = GameState(frame_index=frame_index, timestamp=captured)
        for m in modules:
            try:
                m.process(frame, state)
            except Exception as exc:                  # noqa: BLE001
                print(f"[measure] {m.name} failed ({exc.__class__.__name__}: {exc})")
        found = recognitions(state, ctl, guard_before)

        for p in state.players:
            sample = body_sample(p, ctl) if p.player_id in history else None
            if sample is not None:
                history[p.player_id].append((captured, *sample))

        dt = 0.0 if last_t is None else captured - last_t
        last_t = captured
        playing = ctl.phase == "playing" and not paused
        both_here = all(pid in ctl.fighters and not ctl.fighters[pid].frozen for pid in (0, 1))
        by_id = {p.player_id: p for p in state.players}
        inputs = {pid: rule_inputs(ctl, pid, by_id.get(pid), captured) for pid in (0, 1)}
        # Identity alone can start a round now, so check pose separately: a try
        # with no skeleton would be a Task 2 failure counted as a Task 4 miss.
        no_skeleton = [pid for pid in (0, 1)
                       if pid not in ctl.fighters or ctl.fighters[pid].last_t is None
                       or captured - ctl.fighters[pid].last_t > SKELETON_S]

        # ---- protocol state machine
        if trial is None and queue:
            trial = dict(queue.pop(0), first={}, latency_ms={}, recognised=[], done=False)
            trial["diag"] = {str(pid): new_diag() for pid in trial["actors"]}
            stage, stage_t = "ready", captured
        if trial is not None and playing and both_here:
            elapsed = captured - stage_t
            if stage == "ready" and no_skeleton:
                stage_t = captured             # wait until pose sees both bodies
            elif stage == "ready" and elapsed >= READY_S:
                stage, stage_t = "go", captured
                trial["go_t"] = captured
                trial["go_frame"] = frame_index
                # where each numbered player stood at GO, to check the mapping later
                trial["x"] = {str(pid): round(ctl.fighters[pid].tx) for pid in (0, 1)}
                trial["sides"] = {str(pid): side(ctl, pid) for pid in (0, 1)}
            elif stage == "go" and elapsed >= GO_S:
                stage, stage_t = "rest", captured
            elif stage == "rest" and elapsed >= REST_S:
                finish(trial, data)
                trial = None
        elif trial is not None:
            if stage == "ready":
                stage_t = captured             # wait, then count down again
            elif stage == "go":
                # the round stopped (K.O., lost player): this trial does not count
                queue.insert(0, redo(trial))
                trial = None
            else:
                # the action itself was done; often it is what caused the K.O.
                finish(trial, data)
                trial = None
        in_idle = trial is None and not queue and idle_left > 0

        # ---- what the rules saw: peaks for the acting players, and every frame
        if trial is not None and stage == "go" and playing:
            for pid in trial["actors"]:
                track(trial["diag"][str(pid)], inputs[pid])
        frames_log.writerow(
            [f"{captured:.3f}", frame_index,
             len(data["trials"]) if trial is not None else "",
             ("idle" if in_idle else stage) if trial is not None or in_idle else "",
             trial["action"] if trial is not None else "",
             " ".join(player_label(p) for p in trial["actors"]) if trial is not None else ""]
            + [round(inputs[p][k], 3) if isinstance(inputs[p][k], float) else inputs[p][k]
               for p in (0, 1) for k in FIELDS])

        # ---- count what was recognised
        pending_latency = []
        if playing:
            data["play_s"] += dt
            if in_idle:
                data["idle_s"] += dt
                idle_left -= dt
            for pid, action in found:
                # a trial's own recognitions only count once the trial is done,
                # so a trial redone with x leaves no trace in the precision
                (trial["recognised"] if trial is not None else data["recognised"]).append(action)
                cued = (trial is not None and stage == "go" and pid in trial["actors"]
                        and str(pid) not in trial["first"])
                if cued:
                    trial["first"][str(pid)] = action
                    if action == trial["action"]:
                        began = onset(history.get(pid, ()), action,
                                      trial["go_t"] - ONSET_EARLY_S, captured)
                        if began is not None:
                            pending_latency.append((pid, began))
                else:
                    where = "idle" if in_idle else (stage if trial is not None else "between")
                    event = {"stage": where, "player": pid, "action": action}
                    (trial.setdefault("false", []) if trial is not None
                     else data["false"]).append(event)

        # ---- draw: the normal game picture, plus the cue box
        renderer.react(state)
        output = renderer.draw(frame, state)
        name_tags(output, ctl)
        if paused:
            banner(output, [("PAUSED", (255, 255, 255)), ("press p to continue", (200, 200, 200))])
        elif not playing or not both_here:
            banner(output, [("WAITING FOR THE ROUND", (0, 200, 255)),
                            ("both players in view; the cues start at FIGHT!", (220, 220, 220))])
        elif trial is not None and stage == "ready" and no_skeleton:
            banner(output, [("WAITING FOR POSE", (0, 200, 255)),
                            (" and ".join(player_label(p) for p in no_skeleton)
                             + " has no skeleton: face the camera, step apart",
                             (220, 220, 220))])
        elif trial is not None:
            n = total - len(queue)
            title = {"ready": f"GET READY: {NAMES[trial['action']]}",
                     "go": f"GO!  {NAMES[trial['action']]}",
                     "rest": "relax"}[stage]
            colour = {"ready": (0, 200, 255), "go": (0, 255, 0), "rest": (200, 200, 200)}[stage]
            banner(output, [(title, colour),
                            (f"{who(trial, ctl)}   trial {n}/{total}   x = redo   q = stop",
                             (230, 230, 230))])
        elif in_idle:
            banner(output, [(f"IDLE: talk, stand, move a little  {idle_left:4.0f} s",
                             (0, 200, 255)), ("no sword actions please", (220, 220, 220))])
        else:
            banner(output, [("DONE", (0, 255, 0)), ("press q to see the results", (230, 230, 230))])

        if not args.headless:
            cv2.imshow(WINDOW, output)
        shown = time.time()
        for pid, began in pending_latency:
            trial["latency_ms"][str(pid)] = 1000 * (shown - began)
        frame_index += 1

        if args.headless:
            if frame_index >= args.headless:
                break
            continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("x") and trial is not None:
            queue.insert(0, redo(trial))
            trial = None

    cap.release()
    if video is not None:
        video.release()
    frames_file.close()
    cv2.destroyAllWindows()
    data["mean_fps"] = frame_index / max(1e-6, time.time() - started)
    path = RESULTS / f"task4_{args.condition}.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    text = report(summarise(data))
    (RESULTS / f"task4_{args.condition}.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print(f"\nSaved task4_{args.condition}.json/.txt/_frames.csv"
          f"{'' if video is None else '/.avi'} in {RESULTS}")
    done = sum(t["done"] for t in data["trials"])
    if done < total:
        print(f"Only {done} of {total} trials were completed.")


if __name__ == "__main__":
    main()
