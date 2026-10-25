# Tests for task 4, control. They drive the module with synthetic stick-figure
# poses, so control can be checked before pose and identity are delivered.
#
#   python tests/test_control.py        (from the game/ folder)
#   python -m pytest tests              (if pytest is installed)
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_types import BBox, GameState, Keypoint, Player  # noqa: E402
from modules import control                              # noqa: E402

FPS = 25
DT = 1.0 / FPS
SW = 60.0                     # shoulder width in pixels
FRAME = np.zeros((360, 640, 3), np.uint8)


def body(cx, sy=200.0, guard=False, wrist=None, conf=0.9):
    """Keypoints of a standing player. wrist=(dx, dy) moves the right wrist,
    relative to the right shoulder, in shoulder widths."""
    k = {
        "nose": (cx, sy - 0.6 * SW),
        "left_shoulder": (cx - SW / 2, sy), "right_shoulder": (cx + SW / 2, sy),
        "left_hip": (cx - SW / 3, sy + 1.5 * SW), "right_hip": (cx + SW / 3, sy + 1.5 * SW),
        "left_wrist": (cx - SW / 2, sy + 1.2 * SW), "right_wrist": (cx + SW / 2, sy + 1.2 * SW),
    }
    if guard:   # wrists crossed in front of the chest
        k["left_wrist"] = (cx + 0.25 * SW, sy + 0.4 * SW)
        k["right_wrist"] = (cx - 0.25 * SW, sy + 0.4 * SW)
    if wrist is not None:
        k["right_wrist"] = (cx + SW / 2 + wrist[0] * SW, sy + wrist[1] * SW)
    return {name: Keypoint(x, y, conf) for name, (x, y) in k.items()}


def as_pose_output(kps):
    """Pixel keypoints -> what pose really writes: normalised to a crop box
    that FOLLOWS the player (centred on the shoulders), plus that box."""
    sx = (kps["left_shoulder"].x + kps["right_shoulder"].x) / 2
    sy = (kps["left_shoulder"].y + kps["right_shoulder"].y) / 2
    box = BBox(sx - 2.0 * SW, sy - 1.5 * SW, sx + 2.0 * SW, sy + 4.0 * SW)
    return box, {n: Keypoint((k.x - box.x0) / box.w, (k.y - box.y0) / box.h,
                             k.confidence) for n, k in kps.items()}


class Sim:
    """Feeds frames to the control module at a fixed frame rate.

    pose_format=True hands control crop-normalised keypoints plus the crop,
    exactly like the real pose module, instead of pixels.
    """

    def __init__(self, pose_format=False):
        self.ctl = control.Module()
        self.t = 1000.0
        self.frame = 0
        self.events: list = []
        self.pose_format = pose_format

    def step(self, p0=None, p1=None, missing=(0, 0)):
        state = GameState(frame_index=self.frame, timestamp=self.t)
        for pid, kps in ((0, p0 if p0 is not None else body(200)),
                         (1, p1 if p1 is not None else body(440))):
            player = Player(player_id=pid, keypoints=kps, frames_missing=missing[pid])
            if self.pose_format:
                player.crop, player.keypoints = as_pose_output(kps)
            state.players.append(player)
        self.ctl.process(FRAME, state)
        self.events += [(e.kind, e.player_id) for e in state.events]
        self.t += DT
        self.frame += 1
        return state

    def idle(self, seconds, **kw):
        for _ in range(int(round(seconds * FPS))):
            state = self.step(**kw)
        return state

    def start_round(self):
        self.idle(control.COUNTDOWN_S + 0.2)
        assert ("round_start", -1) in self.events, self.events
        assert self.ctl.phase == "playing"
        self.events.clear()

    def wind_up(self, to, p1=None, seconds=0.8):
        """Player 0 raises the right wrist SLOWLY from rest to `to`."""
        start = (0.0, 1.2)
        n = int(seconds * FPS)
        for i in range(1, n + 1):
            a = i / n
            self.step(p0=body(200, wrist=(start[0] + a * (to[0] - start[0]),
                                          start[1] + a * (to[1] - start[1]))), p1=p1)

    def swing(self, direction="horizontal", p1=None, frames=4, p1_after=None):
        """Player 0 winds up, swings the right wrist fast, then holds still."""
        def wrist(d):
            return (0.3 + d, 0.6) if direction == "horizontal" else (0.6, -0.5 + d)
        self.wind_up(wrist(0), p1=p1)
        for i in range(1, frames + 1):
            self.step(p0=body(200, wrist=wrist(0.6 * i)), p1=p1)
        self.idle(control.RESOLVE_DELAY_S + 0.1, p0=body(200), p1=p1_after or p1)

    def count(self, kind, pid=None):
        return sum(1 for k, p in self.events if k == kind and (pid is None or p == pid))


# ----------------------------------------------------------------------- tests
def test_waits_for_two_players_then_counts_down():
    sim = Sim()
    state = sim.step(missing=(0, 99))
    assert state.message == "WAITING FOR 2 PLAYERS"
    state = sim.step()
    assert state.message.startswith("ROUND 1")
    sim.start_round()


def test_unanswered_swing_does_damage():
    sim = Sim()
    sim.start_round()
    sim.swing("horizontal")
    assert sim.count("attack", 0) == 1
    assert sim.count("damage", 1) == 1, sim.events     # damage names the victim
    assert sim.ctl.fighters[1].health < control.MAX_HEALTH
    assert sim.ctl.fighters[0].health == control.MAX_HEALTH


def test_guard_blocks_vertical_cut():
    sim = Sim()
    sim.start_round()
    guard = body(440, guard=True)
    sim.idle(0.2, p1=guard)
    assert sim.count("block", 1) > 0
    sim.swing("vertical", p1=guard)
    assert sim.count("blocked", 1) == 1, sim.events
    assert sim.count("damage") == 0
    assert sim.ctl.fighters[1].health == control.MAX_HEALTH


def test_duck_dodges_horizontal_cut():
    sim = Sim()
    sim.start_round()
    duck = body(440, sy=200 + 0.7 * SW)
    sim.swing("horizontal", p1=duck)
    assert sim.count("dodge", 1) >= 1
    assert sim.count("missed", 0) == 1, sim.events     # missed names the attacker
    assert sim.count("damage") == 0


def test_duck_does_not_save_from_vertical_cut():
    sim = Sim()
    sim.start_round()
    sim.swing("vertical", p1=body(440, sy=200 + 0.7 * SW))
    assert sim.count("damage", 1) == 1, sim.events


def test_sidestep_dodges_vertical_cut():
    sim = Sim()
    sim.start_round()
    sim.wind_up((0.6, -0.5))
    for i in range(1, 5):                               # cut, while P1 steps right
        sim.step(p0=body(200, wrist=(0.6, -0.5 + 0.6 * i)), p1=body(440 + 18 * i))
    sim.idle(control.RESOLVE_DELAY_S + 0.1, p1=body(512))
    assert sim.count("missed", 0) == 1, sim.events
    assert sim.count("damage") == 0


def test_simultaneous_swings_clash():
    sim = Sim()
    sim.start_round()

    def p1_wrist(dx):           # P1 faces left: its image-left wrist swings left
        b = body(440)
        b["left_wrist"] = Keypoint(440 - SW / 2 - dx * SW, 200 + 0.6 * SW, 0.9)
        return b
    n = int(0.8 * FPS)
    for i in range(1, n + 1):   # both wind up slowly
        a = i / n
        sim.step(p0=body(200, wrist=(0.3 * a, 1.2 - 0.6 * a)), p1=p1_wrist(0.3 * a))
    for i in range(1, 5):       # and swing on the same frames
        sim.step(p0=body(200, wrist=(0.3 + 0.6 * i, 0.6)), p1=p1_wrist(0.3 + 0.6 * i))
    sim.idle(control.RESOLVE_DELAY_S + 0.1)
    assert sim.count("attack") == 2, sim.events
    assert sim.count("blocked", 0) == 1 and sim.count("blocked", 1) == 1, sim.events
    assert sim.count("damage") == 0


def test_effects_are_drawn_on_the_players():
    sim = Sim()
    sim.start_round()
    elbows = {"right_elbow": Keypoint(200 + SW / 2, 200 + 0.6 * SW, 0.9),
              "left_elbow": Keypoint(200 - SW / 2, 200 + 0.6 * SW, 0.9)}
    state = sim.step(p0={**body(200), **elbows}, p1=body(440, guard=True))
    lines = [c for c in state.draw if c.kind == "line"]
    circles = [c for c in state.draw if c.kind == "circle"]
    assert lines, "katana missing"
    sim.idle(0.2, p1=body(440, guard=True))
    state = sim.step(p1=body(440, guard=True))
    assert any(c.kind == "circle" and abs(c.x - 440) < SW for c in state.draw), \
        "guard shield not on player 1"


def test_walking_and_fidgeting_do_not_attack():
    sim = Sim()
    sim.start_round()
    for i in range(3 * FPS):
        cx = 200 + 40 * np.sin(i / 10)                 # slow sway sideways
        jitter = np.random.default_rng(i).normal(0, 2, 2)
        sim.step(p0=body(cx, wrist=(0.0 + jitter[0] / SW, 1.2 + jitter[1] / SW)))
    assert sim.count("attack") == 0, sim.events


def test_raising_guard_fast_is_not_an_attack():
    sim = Sim()
    sim.start_round()
    sim.idle(0.5, p1=body(440, guard=True))             # snaps straight into guard
    assert sim.count("attack") == 0, sim.events
    assert sim.count("block", 1) > 0


def test_dropping_guard_fast_is_not_an_attack():
    sim = Sim()
    sim.start_round()
    sim.idle(0.5, p0=body(200, guard=True))
    sim.idle(0.5, p0=body(200))                         # arms snap back down
    assert sim.count("attack") == 0, sim.events


def test_holding_a_duck_announces_one_dodge():
    sim = Sim()
    sim.start_round()
    sim.idle(1.5, p1=body(440, sy=200 + 0.7 * SW))
    assert sim.count("dodge", 1) == 1, sim.events


def test_low_confidence_keypoints_trigger_nothing():
    sim = Sim()
    sim.start_round()
    for i in range(4):
        sim.step(p0=body(200, wrist=(0.3 + 0.6 * i, 0.6), conf=0.2))
    assert sim.count("attack") == 0


def test_cooldown_allows_one_attack_per_swing():
    sim = Sim()
    sim.start_round()
    for i in range(10):                                 # one long fast swing
        sim.step(p0=body(200, wrist=(0.3 + 0.4 * i, 0.6)))
    assert sim.count("attack", 0) == 1


def test_lost_player_is_frozen_and_cannot_be_hit():
    sim = Sim()
    sim.start_round()
    state = sim.idle(0.4, missing=(0, control.FREEZE_AFTER + 1))
    assert "P1 LOST" in state.message
    sim.swing("horizontal")
    assert sim.count("damage") == 0
    # After coming back there is a grace period, then they can be hit again.
    sim.idle(control.RETURN_GRACE_S + 0.1)
    sim.swing("horizontal")
    assert sim.count("damage", 1) == 1


def test_knockout_ends_round_and_next_round_restores_health():
    sim = Sim()
    sim.start_round()
    sim.ctl.fighters[1].health = 5
    sim.swing("horizontal")
    assert sim.count("ko", 1) == 1
    assert sim.ctl.phase == "over"
    assert sim.ctl.fighters[0].rounds == 1
    state = sim.step()
    assert state.message == "P0 WINS ROUND 1" and state.players[0].score == 1
    sim.idle(control.ROUND_OVER_S)
    sim.start_round()
    assert sim.ctl.fighters[1].health == control.MAX_HEALTH


def test_only_known_events_are_raised():
    from game_types import EVENTS
    sim = Sim()
    sim.start_round()
    sim.swing("horizontal")
    sim.swing("vertical", p1=body(440, guard=True))
    assert all(kind in EVENTS for kind, _ in sim.events)


def test_round_waits_until_pose_sees_both_bodies():
    sim = Sim()
    state = sim.idle(1.0, p1={})                        # identity box, no skeleton
    assert state.message == "WAITING FOR 2 PLAYERS", state.message
    sim.start_round()


# --------------------------------------------- with the real pose output format
def test_pose_format_swing_damages():
    sim = Sim(pose_format=True)
    sim.start_round()
    sim.swing("horizontal")
    assert sim.count("damage", 1) == 1, sim.events


def test_pose_format_sidestep_and_duck_still_detected():
    # The crop follows the body, so in crop units the torso barely moves.
    # Only after converting to pixels can a dodge be seen.
    sim = Sim(pose_format=True)
    sim.start_round()
    for i in range(1, 6):
        sim.step(p1=body(440 + 18 * i))
    assert sim.count("dodge", 1) == 1, sim.events
    sim.idle(1.0, p1=body(530))
    sim.idle(0.5, p1=body(530, sy=200 + 0.7 * SW))
    assert sim.count("dodge", 1) == 2, sim.events


def test_pose_format_guard_blocks():
    sim = Sim(pose_format=True)
    sim.start_round()
    guard = body(440, guard=True)
    sim.swing("vertical", p1=guard)
    assert sim.count("blocked", 1) == 1, sim.events


def test_identity_swap_resets_the_body():
    # Identity hands id 0 a different body (players crossed). The jump must
    # not count as a swing, and the new body is not "ducking" just because
    # it is shorter than the old one.
    sim = Sim()
    sim.start_round()
    sim.idle(0.5, p0=body(200))
    sim.idle(1.0, p0=body(330, sy=200 + 0.8 * SW))     # another, shorter person
    assert sim.count("attack") == 0, sim.events
    assert not sim.ctl.fighters[0].ducking


def test_short_tracking_gap_gives_a_grace_period():
    sim = Sim()
    sim.start_round()
    sim.idle(0.2, missing=(2, 0))                       # lost for a few frames only
    sim.wind_up((0.3, 0.6), seconds=0.2)                # swings right after return
    for i in range(1, 5):
        sim.step(p0=body(200, wrist=(0.3 + 0.6 * i, 0.6)))
    assert sim.count("attack") == 0, sim.events


def test_no_attack_while_partly_outside_the_frame():
    sim = Sim()
    sim.start_round()
    sim.wind_up((0.3, 0.6))
    for i in range(1, 5):
        state = GameState(frame_index=sim.frame, timestamp=sim.t)
        state.players = [
            Player(0, keypoints=body(200, wrist=(0.3 + 0.6 * i, 0.6)),
                   bbox=BBox(0, 100, 260, 360)),        # touches the left edge
            Player(1, keypoints=body(440)),
        ]
        sim.ctl.process(FRAME, state)
        sim.events += [(e.kind, e.player_id) for e in state.events]
        sim.t += DT
        sim.frame += 1
    assert sim.count("attack") == 0, sim.events


def test_guessed_wrist_cannot_attack():
    # Pose fills hidden wrists from memory/template at 0.58-0.70 confidence.
    sim = Sim()
    sim.start_round()
    sim.wind_up((0.3, 0.6))
    for i in range(1, 5):
        kps = body(200, wrist=(0.3 + 0.6 * i, 0.6))
        kps["right_wrist"].confidence = 0.65
        sim.step(p0=kps)
    assert sim.count("attack") == 0, sim.events


def test_within_time_budget():
    from game_types import BUDGET_MS
    sim = Sim()
    sim.start_round()
    started = time.perf_counter()
    n = 500
    for i in range(n):
        sim.step(p0=body(200, wrist=(0.3 + 0.6 * (i % 4), 0.6)))
    ms = 1000 * (time.perf_counter() - started) / n
    print(f"    control: {ms:.3f} ms per frame (budget {BUDGET_MS['control']} ms)")
    assert ms < BUDGET_MS["control"]


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}  {exc}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
