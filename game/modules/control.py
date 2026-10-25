# TASK 4 - player interaction and game control.
#
# IPCV group project. Owner: Vaishnav (s number).
#
# WHAT THIS MODULE DOES
#   Pose (task 2) tells me where everyone's joints are. Identity (task 3) tells
#   me who is who. I turn that into a game:
#     - recognise ACTIONS from the keypoints: attack, guard, dodge
#     - RESOLVE every attack against what the defender did: damage, blocked, missed
#     - keep health and rounds, and run the round flow: waiting, countdown,
#       fight, KO, next round
#     - raise the events in game_types.EVENTS so task 5 can show them
#     - draw the BODY-ALIGNED effects of those actions: a katana along the
#       forearm, the slash arc, the guard shield, hit sparks and damage numbers.
#       Task 5 reacts with the whole scene (shake, flash); these sit on the player.
#
# CONFLICTING ACTIONS
#   - Guarding and attacking at once: the guard wins, no attack is possible.
#   - Both players swing within CLASH_WINDOW_S of each other: the blades meet,
#     both swings are cancelled and both players get a "blocked".
#   - Swings further apart than that are resolved independently, so a trade
#     of hits is possible, which is how a duel works.
#
# WHY IT KEEPS ITS OWN MEMORY
#   main.py builds a brand new GameState every frame, so anything written to a
#   Player is gone on the next frame. Health, rounds and motion history live
#   here, keyed by player_id, and are copied back into the players each frame.
#
# DESIGN CHOICES (the "not decided yet" items from the task 5 requirements)
#   - Everything is measured in SHOULDER WIDTHS, not pixels. A player close to
#     the camera and one far away then need the same swing to attack.
#   - Wrist speed is measured RELATIVE TO THE SHOULDERS. Walking or ducking
#     moves the whole body, which must not count as a swing.
#   - Attacks are resolved by TIMING, not reach, because one webcam cannot see
#     depth. A swing gives the defender RESOLVE_DELAY_S to guard or dodge.
#   - Which defence stops which attack follows docs/game_flow.md: crossed arms
#     stop a vertical cut, ducking avoids a horizontal one, a sidestep avoids both.
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field, replace

from game_types import EVENTS, DrawCommand, GameEvent, GameState, Keypoint, Player
from game_types import Module as Base

# ------------------------------------------------------------------ tuning
# Body-relative units: 1.0 = one shoulder width. Times are in seconds.
FREEZE_AFTER = 5            # frames_missing above this freezes a player (task 3)
RETURN_GRACE_S = 1.0        # after coming back: cannot attack or be hit
REACQUIRE_GRACE_S = 0.5     # after any tracking gap or body jump: cannot attack
TELEPORT_SHIFT = 1.0        # torso moving this far in one frame is not a person
                            # moving, it is identity handing us another body
EDGE_PX = 3                 # a box this close to the side of the frame is a body
                            # partly outside it: keypoints unreliable, no attacks

ATTACK_SPEED = 4.0          # wrist speed that counts as a swing, widths/s
ATTACK_SPEED_MAX = 10.0     # speed that gives full strength
ATTACK_COOLDOWN_S = 0.7     # one swing per player per this long
MIN_REACH = 0.7             # wrist this far from the chest, ignores fidgeting
ATTACK_MIN_CONF = 0.75      # only a wrist pose actually SAW may swing. Pose fills
                            # gaps with guessed wrists (template 0.58, memory 0.70,
                            # predicted decaying), fine to draw, not to attack with
SPEED_WINDOW_S = 0.10       # velocity is measured over this much history
MAX_GAP_S = 0.25            # older reference than this means tracking dropped

RESOLVE_DELAY_S = 0.35      # time the defender has to react to a swing
DEFENCE_EARLY_S = 0.20      # a defence started this long before the swing counts
CLASH_WINDOW_S = 0.15       # two swings this close together cancel each other

GUARD_ON_FRAMES = 3         # crossed arms needed for this many frames to guard
GUARD_OFF_FRAMES = 3        # and uncrossed this many to drop the guard
GUARD_RELEASE_S = 0.3       # no attack this soon after the guard drops: dropping
                            # the arms fast looks just like a swing
DUCK_DROP = 0.45            # shoulders below standing height by this much
STEP_SHIFT = 0.8            # torso moved sideways this much ...
STEP_WINDOW_S = 0.30        # ... within this long
DODGE_REPEAT_S = 0.40       # at most one dodge event this often
BASELINE_TAU_S = 2.0        # how fast standing height adapts while standing
BASELINE_TAU_DUCK_S = 6.0   # and while ducking, so nobody can camp in a duck

MAX_HEALTH = 100
BASE_DAMAGE = 8             # damage = BASE_DAMAGE + EXTRA_DAMAGE * strength
EXTRA_DAMAGE = 12
COUNTDOWN_S = 3.0
FIGHT_MESSAGE_S = 0.7
ROUND_OVER_S = 3.0

# Which defences beat which attack direction (see docs/game_flow.md).
DEFENCES = {
    "horizontal": ("duck", "step"),
    "vertical": ("guard", "step"),
}

DRAW_HUD = True             # health bars and round counter
DRAW_FX = True              # katana, slash arcs, guard shield, sparks, damage numbers
DRAW_DEBUG = True           # per-player action labels (GUARD, DUCK, STEP)

# Used when identity has not given a player a colour. BGR: P0 blue, P1 red.
PALETTE = ((255, 140, 60), (60, 80, 255))
STEEL = (225, 225, 235)
HIT = (40, 40, 255)
SPARK = (255, 255, 255)
GUARD_GLOW = (255, 220, 120)

# Set IPCV_CONTROL_LOG=events.csv to log every event, for the report metrics.
LOG_PATH = os.environ.get("IPCV_CONTROL_LOG", "")

NEVER = -1e9


@dataclass
class Fighter:
    """What I remember about one player between frames."""
    player_id: int
    health: int = MAX_HEALTH
    rounds: int = 0
    scale: float = 0.0                      # smoothed shoulder width in pixels
    wrists: dict = field(default_factory=lambda: {"left_wrist": deque(maxlen=30),
                                                  "right_wrist": deque(maxlen=30)})
    torso: deque = field(default_factory=lambda: deque(maxlen=30))   # (t, x, y)
    baseline_y: float | None = None         # standing shoulder height
    last_t: float | None = None

    guard: bool = False
    guard_streak: int = 0                   # frames the raw signal disagreed
    ducking: bool = False
    dodging: bool = False                   # ducking or mid-sidestep, last frame
    last_guard_t: float = NEVER
    last_duck_t: float = NEVER
    last_step_t: float = NEVER
    last_dodge_event_t: float = NEVER
    last_attack_t: float = NEVER

    frozen: bool = False
    active_from: float = NEVER              # end of the grace period after a return
    gap: bool = False                       # missing for at least one frame lately
    at_edge: bool = False                   # body partly outside the frame

    # Where things are on screen, for drawing effects on the right body part.
    tx: float = 0.0                         # shoulder midpoint, latest frame
    ty: float = 0.0
    sword_hand: str = "right_wrist"         # the wrist that swung last holds the katana
    paths: dict = field(default_factory=lambda: {"left_wrist": deque(maxlen=8),
                                                 "right_wrist": deque(maxlen=8)})

    def forget_motion(self) -> None:
        for hist in (*self.wrists.values(), *self.paths.values()):
            hist.clear()
        self.torso.clear()
        self.last_t = None
        self.guard, self.guard_streak, self.ducking, self.dodging = False, 0, False, False
        # Size and standing height belong to a body. After a gap it may be a
        # different body, so learn them again rather than see a "duck".
        self.scale, self.baseline_y = 0.0, None


@dataclass
class Attack:
    attacker: int
    defender: int
    t: float
    strength: float
    direction: str                          # "horizontal" or "vertical"
    x: float = 0.0                          # where the blade was, for the clash spark
    y: float = 0.0


@dataclass
class Effect:
    """A short-lived visual effect, drawn every frame until it expires."""
    kind: str                               # slash, spark, text, shield, ghost
    t0: float
    duration: float
    x: float
    y: float
    size: float                             # usually one shoulder width
    colour: tuple
    text: str = ""
    points: tuple = ()                      # the wrist path, for a slash


class Module(Base):
    name = "control"

    def __init__(self):
        self.reset()
        self._log = None

    def reset(self) -> None:
        self.fighters: dict[int, Fighter] = {}
        self.pending: list[Attack] = []
        self.fx: list[Effect] = []
        self.phase = "waiting"             # waiting, countdown, playing, over
        self.phase_t = 0.0                  # when the current phase started
        self.winner: int | None = None
        self.round = 0

    # ============================================================ main entry
    def process(self, frame, state: GameState) -> None:
        t = state.timestamp
        height, width = frame.shape[:2]

        self._sync_players(state, t)
        playing = self.phase == "playing"

        # Pose gives crop-normalised keypoints. Everything below works in
        # frame pixels, so convert once here. These copies are only read;
        # health and score are written to the real players further down.
        players = [_in_pixels(p) for p in state.players]

        for player in players:
            fighter = self.fighters[player.player_id]
            if fighter.frozen:
                continue
            # identity's box, not pose's crop: the crop is padded and clamped
            # to the frame, so it touches the edge even when the body does not
            box = player.bbox
            fighter.at_edge = box is not None and (box.x0 <= EDGE_PX
                                                   or box.x1 >= width - EDGE_PX)
            self._sense(fighter, player, t, state, emit=playing)

        if playing:
            self._resolve(state, t)
        self._update_round(state, t)

        for player in state.players:
            fighter = self.fighters[player.player_id]
            player.health = fighter.health
            player.score = fighter.rounds

        if DRAW_FX:
            self._draw_effects(state, players, t)
        if DRAW_HUD:
            self._draw_hud(state, width, height)
        if DRAW_DEBUG:
            self._draw_labels(state, players, t)

    # ========================================================== bookkeeping
    def _sync_players(self, state: GameState, t: float) -> None:
        """Create fighters for new ids and freeze or thaw lost players."""
        seen = set()
        for player in state.players:
            seen.add(player.player_id)
            fighter = self.fighters.get(player.player_id)
            if fighter is None:
                fighter = self.fighters[player.player_id] = Fighter(player.player_id)
            lost = player.frames_missing > FREEZE_AFTER
            if lost and not fighter.frozen:
                fighter.frozen = True
                fighter.forget_motion()
            elif not lost and fighter.frozen:
                fighter.frozen = False
                fighter.active_from = t + RETURN_GRACE_S
            # Even a short gap can hide an identity swap (two players crossing),
            # so whoever comes back starts with a clean history.
            if player.frames_missing > 0:
                if not fighter.gap:
                    fighter.gap = True
                    fighter.forget_motion()
            elif fighter.gap:
                fighter.gap = False
                fighter.active_from = max(fighter.active_from, t + REACQUIRE_GRACE_S)
        # A player that identity dropped from the list entirely is also lost.
        for pid, fighter in self.fighters.items():
            if pid not in seen and not fighter.frozen:
                fighter.frozen = True
                fighter.forget_motion()

    def _emit(self, state: GameState, kind: str, player_id: int,
              strength: float = 1.0, log: bool = True) -> None:
        assert kind in EVENTS, f"unknown event {kind}"
        state.events.append(GameEvent(kind=kind, player_id=player_id, strength=strength))
        if log and LOG_PATH:
            self._write_log(state, kind, player_id, strength)

    def _write_log(self, state, kind, player_id, strength) -> None:
        if self._log is None:
            new = not os.path.exists(LOG_PATH)
            self._log = open(LOG_PATH, "a", encoding="utf-8")
            if new:
                self._log.write("timestamp,frame,event,player_id,strength\n")
        self._log.write(f"{state.timestamp:.3f},{state.frame_index},{kind},"
                        f"{player_id},{strength:.2f}\n")
        self._log.flush()

    def _active(self, fighter: Fighter, t: float) -> bool:
        return not fighter.frozen and t >= fighter.active_from

    def _opponent(self, player_id: int) -> Fighter | None:
        for pid, fighter in self.fighters.items():
            if pid != player_id:
                return fighter
        return None

    # ============================================================== sensing
    def _sense(self, f: Fighter, p: Player, t: float, state: GameState,
               emit: bool) -> None:
        """Update one player's motion features and detect their actions."""
        ls, rs = p.point("left_shoulder"), p.point("right_shoulder")
        if ls is None or rs is None:
            return                          # no body frame, no decisions
        tx, ty = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2

        # Body scale. Shoulder width shrinks when a player turns sideways, so
        # torso length (about 1.5 shoulder widths) keeps it honest.
        scale = math.hypot(rs.x - ls.x, rs.y - ls.y)
        lh, rh = p.point("left_hip"), p.point("right_hip")
        if lh is not None and rh is not None:
            torso_len = math.hypot((lh.x + rh.x) / 2 - tx, (lh.y + rh.y) / 2 - ty)
            scale = max(scale, torso_len / 1.5)
        if scale < 5:
            return

        # A torso cannot cover a shoulder width in one frame. If it "did",
        # identity gave this id to another body: start that body from scratch.
        if (f.torso and f.scale > 0 and t - f.torso[-1][0] < MAX_GAP_S
                and math.hypot(tx - f.torso[-1][1], ty - f.torso[-1][2])
                > TELEPORT_SHIFT * f.scale):
            f.forget_motion()
            f.active_from = max(f.active_from, t + REACQUIRE_GRACE_S)

        dt = 0.0 if f.last_t is None else max(0.0, t - f.last_t)
        f.last_t = t
        f.scale = scale if f.scale == 0 else 0.8 * f.scale + 0.2 * scale
        s = f.scale

        f.torso.append((t, tx, ty))
        _trim(f.torso, t)
        f.tx, f.ty = tx, ty

        # ---- guard: wrists crossed in front of the chest, with hysteresis
        lw, rw = p.point("left_wrist"), p.point("right_wrist")
        raw_guard = False
        if lw is not None and rw is not None:
            crossed = (rw.x - lw.x) * (rs.x - ls.x) < 0
            chest = all(ty - 0.5 * s < w.y < ty + 1.6 * s for w in (lw, rw))
            close = math.hypot(lw.x - rw.x, lw.y - rw.y) < 1.2 * s
            raw_guard = crossed and chest and close
        guard_raised = False
        if raw_guard != f.guard:
            f.guard_streak += 1
            if f.guard_streak >= (GUARD_ON_FRAMES if raw_guard else GUARD_OFF_FRAMES):
                f.guard, f.guard_streak = raw_guard, 0
                guard_raised = raw_guard
        else:
            f.guard_streak = 0
        if f.guard:
            f.last_guard_t = t
            if emit:
                # Sent every frame while held, like a state, so the scene can
                # keep the guard glow on. Only the frame it went up is logged.
                self._emit(state, "block", p.player_id, log=guard_raised)

        # ---- duck: shoulders dropped well below their standing height
        if f.baseline_y is None:
            f.baseline_y = ty
        f.ducking = ty - f.baseline_y > DUCK_DROP * s
        tau = BASELINE_TAU_DUCK_S if f.ducking else BASELINE_TAU_S
        f.baseline_y += (ty - f.baseline_y) * (1 - math.exp(-dt / tau))
        if f.ducking:
            f.last_duck_t = t

        # ---- sidestep: torso moved sideways quickly
        stepped = False
        for t0, x0, _ in f.torso:
            if t - t0 <= STEP_WINDOW_S and abs(tx - x0) > STEP_SHIFT * s:
                stepped = True
                break
        if stepped:
            f.last_step_t = t

        # Announced once when the dodge starts, not again while it is held.
        dodge_started = (f.ducking or stepped) and not f.dodging
        f.dodging = f.ducking or stepped
        if dodge_started and t - f.last_dodge_event_t > DODGE_REPEAT_S:
            f.last_dodge_event_t = t
            if emit:
                self._emit(state, "dodge", p.player_id)
                # An afterimage where the body was a moment ago.
                old_x, old_y = f.torso[0][1], f.torso[0][2]
                if f.ducking:
                    old_y = f.baseline_y
                self._effect("ghost", t, 0.35, old_x, old_y, s, _colour(p))

        # ---- attack: a wrist moving fast relative to the shoulders
        best = None
        for name, wrist in (("left_wrist", lw), ("right_wrist", rw)):
            hist = f.wrists[name]
            if wrist is None:
                hist.clear()                # a gap must not look like a jump
                continue
            hist.append((t, wrist.x - tx, wrist.y - ty))
            _trim(hist, t)
            f.paths[name].append((wrist.x, wrist.y))
            v = _velocity(hist)
            if v is None:
                continue
            speed = math.hypot(*v) / s
            rx, ry = wrist.x - tx, wrist.y - ty
            reach = math.hypot(rx, ry) / s
            # A strike moves the hand AWAY from the body. Pulling the arms in
            # to cross a guard is just as fast, but it points inward.
            outward = rx * v[0] + ry * v[1] > 0
            if (speed >= ATTACK_SPEED and reach >= MIN_REACH and outward
                    and wrist.confidence >= ATTACK_MIN_CONF):
                if best is None or speed > best[0]:
                    best = (speed, v, wrist, name)

        if (best is not None and not f.guard and not f.at_edge and self._active(f, t)
                and t - f.last_guard_t > GUARD_RELEASE_S
                and t - f.last_attack_t >= ATTACK_COOLDOWN_S):
            speed, (vx, vy), wrist, hand = best
            f.last_attack_t = t
            f.sword_hand = hand
            strength = 0.3 + 0.7 * min(1.0, (speed - ATTACK_SPEED)
                                       / (ATTACK_SPEED_MAX - ATTACK_SPEED))
            direction = "horizontal" if abs(vx) >= abs(vy) else "vertical"
            if emit:
                self._emit(state, "attack", p.player_id, strength)
                # The slash arc follows the path the wrist actually took, then
                # carries on a little in the direction of the swing, like a blade.
                tip = (wrist.x + vx * 0.08, wrist.y + vy * 0.08)
                self._effect("slash", t, 0.35, wrist.x, wrist.y, s, _colour(p),
                             points=(*f.paths[hand], tip))
                target = self._opponent(p.player_id)
                if target is not None and self._active(target, t):
                    self._queue_attack(state, Attack(p.player_id, target.player_id, t,
                                                     strength, direction,
                                                     wrist.x, wrist.y), s)

    def _queue_attack(self, state: GameState, atk: Attack, s: float) -> None:
        """Queue a swing, unless the opponent swung at the same moment."""
        for other in self.pending:
            if other.attacker == atk.defender and atk.t - other.t <= CLASH_WINDOW_S:
                # Blades meet. Neither swing lands, both players see the spark.
                self.pending.remove(other)
                strength = max(atk.strength, other.strength)
                self._emit(state, "blocked", atk.attacker, strength)
                self._emit(state, "blocked", atk.defender, strength)
                x, y = (atk.x + other.x) / 2, (atk.y + other.y) / 2
                self._effect("spark", atk.t, 0.4, x, y, 1.6 * s, SPARK)
                self._effect("text", atk.t, 0.8, x, y - s, s, SPARK, text="CLASH!")
                return
        self.pending.append(atk)

    def _effect(self, kind, t, duration, x, y, size, colour, text="", points=()) -> None:
        self.fx.append(Effect(kind, t, duration, x, y, size, colour, text, tuple(points)))

    # =========================================================== resolution
    def _resolve(self, state: GameState, t: float) -> None:
        """Decide every swing in flight: blocked, missed, or a hit."""
        still_pending = []
        for atk in self.pending:
            attacker = self.fighters[atk.attacker]
            defender = self.fighters[atk.defender]
            if attacker.frozen or not self._active(defender, t):
                continue                    # someone left, the swing is void
            since = atk.t - DEFENCE_EARLY_S
            works = DEFENCES[atk.direction]
            s = defender.scale or 50.0
            chest = (defender.tx, defender.ty + 0.5 * s)
            if "guard" in works and defender.last_guard_t >= since:
                self._emit(state, "blocked", atk.defender, atk.strength)
                self._effect("spark", t, 0.35, *chest, 1.3 * s, SPARK)
                self._effect("text", t, 0.8, chest[0], chest[1] - 2.4 * s, s,
                             GUARD_GLOW, text="BLOCKED")
            elif (("duck" in works and defender.last_duck_t >= since)
                  or ("step" in works and defender.last_step_t >= since)):
                self._emit(state, "missed", atk.attacker, atk.strength)
                self._effect("text", t, 0.8, atk.x, atk.y - s, s, (190, 190, 190),
                             text="MISS")
            elif t - atk.t >= RESOLVE_DELAY_S:
                damage = round(BASE_DAMAGE + EXTRA_DAMAGE * atk.strength)
                defender.health = max(0, defender.health - damage)
                self._emit(state, "damage", atk.defender, atk.strength)
                self._effect("spark", t, 0.45, *chest, 1.8 * s, HIT)
                self._effect("text", t, 1.0, chest[0], chest[1] - 2.4 * s, 1.3 * s,
                             HIT, text=f"-{damage}")
                if defender.health == 0:
                    self._knockout(state, t, loser=defender, winner=attacker)
                    return
            else:
                still_pending.append(atk)
        self.pending = still_pending

    def _knockout(self, state, t, loser: Fighter, winner: Fighter) -> None:
        winner.rounds += 1
        self.winner = winner.player_id
        self.pending.clear()
        self._emit(state, "ko", loser.player_id)
        self.fx = [e for e in self.fx if e.kind != "text"]   # K.O. stands alone
        s = loser.scale or 50.0
        self._effect("spark", t, 0.8, loser.tx, loser.ty + 0.5 * s, 3 * s, HIT)
        self._effect("text", t, ROUND_OVER_S, loser.tx, loser.ty - 1.2 * s, 2 * s,
                     HIT, text="K.O.")
        self._set_phase("over", t)

    # ================================================================ rounds
    def _set_phase(self, phase: str, t: float) -> None:
        self.phase, self.phase_t = phase, t

    def _update_round(self, state: GameState, t: float) -> None:
        # Ready = identity tracks them AND pose has seen their body lately.
        # A box with no skeleton in it cannot fight, so do not start a round.
        ready = [f for f in self.fighters.values()
                 if not f.frozen and f.last_t is not None and t - f.last_t < 1.0]
        two_ready = len(ready) >= 2
        elapsed = t - self.phase_t

        if self.phase == "waiting":
            if two_ready:
                self._set_phase("countdown", t)
                elapsed = 0.0
            else:
                state.phase, state.message = "waiting", "WAITING FOR 2 PLAYERS"
                return

        if self.phase == "countdown":
            if not two_ready:
                self._set_phase("waiting", t)
                state.phase, state.message = "waiting", "WAITING FOR 2 PLAYERS"
                return
            if elapsed < COUNTDOWN_S:
                state.phase = "waiting"
                state.message = f"ROUND {self.round + 1}   {math.ceil(COUNTDOWN_S - elapsed)}"
                return
            self.round += 1
            for f in self.fighters.values():
                f.health = MAX_HEALTH
            self.pending.clear()
            self._set_phase("playing", t)
            self._emit(state, "round_start", -1)
            elapsed = 0.0

        if self.phase == "playing":
            state.phase = "playing"
            lost = [f.player_id for f in self.fighters.values() if f.frozen]
            if lost:
                state.message = f"P{lost[0]} LOST - STEP BACK IN"
            elif elapsed < FIGHT_MESSAGE_S:
                state.message = "FIGHT!"
            return

        if self.phase == "over":
            state.phase = "over"
            state.message = f"P{self.winner} WINS ROUND {self.round}"
            if elapsed >= ROUND_OVER_S:
                self._set_phase("countdown" if two_ready else "waiting", t)

    # ============================================================== drawing
    def _draw_hud(self, state: GameState, width: int, height: int) -> None:
        """Health bars along the top, P0 on the left, P1 on the right."""
        colours = {p.player_id: _colour(p) for p in state.players}
        bar_w, bar_h = int(width * 0.33), max(8, int(height * 0.035))
        y = int(height * 0.04)
        for pid in sorted(self.fighters)[:2]:
            f = self.fighters[pid]
            left = pid == min(self.fighters)
            x0 = int(width * 0.07) if left else width - int(width * 0.07) - bar_w
            fill = int(bar_w * f.health / MAX_HEALTH)
            colour = colours.get(pid, (255, 255, 255))
            draw = state.draw.append
            draw(DrawCommand("rect", x0, y, x0 + bar_w, y + bar_h,
                             colour=(40, 40, 40), thickness=-1, layer=2))
            if fill > 0:
                fx0 = x0 if left else x0 + bar_w - fill   # bars drain toward the centre
                draw(DrawCommand("rect", fx0, y, fx0 + fill, y + bar_h,
                                 colour=_health_colour(f.health), thickness=-1, layer=2))
            draw(DrawCommand("rect", x0, y, x0 + bar_w, y + bar_h,
                             colour=colour, thickness=1, layer=2))
            label = f"P{pid}  {f.health}  wins {f.rounds}"
            if f.frozen:
                label += "  LOST"
            draw(DrawCommand("text", x0, y + bar_h + 16, text=label,
                             colour=(235, 235, 235), thickness=1, layer=2, scale=0.45))

    def _draw_labels(self, state: GameState, players: list[Player], t: float) -> None:
        """Which action each player is doing right now, above their head."""
        for p in players:
            f = self.fighters[p.player_id]
            if f.frozen or f.scale == 0:
                continue
            tags = [name for name, on in (("GUARD", f.guard), ("DUCK", f.ducking),
                                          ("STEP", t - f.last_step_t < 0.3)) if on]
            if t < f.active_from:
                tags.append("READY?")
            if tags:
                # Above the head: pose has no nose point, so go up from the
                # shoulders by about a head and a half.
                state.draw.append(DrawCommand("text", f.tx - 30, f.ty - 1.6 * f.scale,
                                              text=" ".join(tags), colour=_colour(p),
                                              thickness=1, layer=1, scale=0.5))

    def _draw_effects(self, state: GameState, players: list[Player], t: float) -> None:
        """Everything that sits ON a player: katana, guard shield, slash, sparks."""
        draw = state.draw.append
        for p in players:
            f = self.fighters[p.player_id]
            if f.frozen or f.scale == 0:
                continue
            s = f.scale

            # The katana, held in the sword hand and pointing along the forearm,
            # so it follows the arm wherever pose puts it.
            elbow = p.point(f.sword_hand.replace("wrist", "elbow"))
            wrist = p.point(f.sword_hand)
            if elbow is not None and wrist is not None:
                dx, dy = wrist.x - elbow.x, wrist.y - elbow.y
                n = math.hypot(dx, dy)
                if n > 1:
                    dx, dy = dx / n, dy / n
                    hilt = (wrist.x - 0.15 * s * dx, wrist.y - 0.15 * s * dy)
                    guard = (wrist.x + 0.2 * s * dx, wrist.y + 0.2 * s * dy)
                    tip = (wrist.x + 1.5 * s * dx, wrist.y + 1.5 * s * dy)
                    draw(DrawCommand("line", *hilt, *guard, colour=_colour(p),
                                     thickness=max(3, int(0.12 * s)), layer=1))
                    draw(DrawCommand("line", guard[0] - 0.12 * s * dy, guard[1] + 0.12 * s * dx,
                                     guard[0] + 0.12 * s * dy, guard[1] - 0.12 * s * dx,
                                     colour=(40, 180, 220), thickness=3, layer=1))
                    draw(DrawCommand("line", *guard, *tip, colour=STEEL,
                                     thickness=max(2, int(0.06 * s)), layer=1))

            # The guard shield: a pulsing ring where the arms cross.
            lw, rw = p.point("left_wrist"), p.point("right_wrist")
            if f.guard and lw is not None and rw is not None:
                cx, cy = (lw.x + rw.x) / 2, (lw.y + rw.y) / 2
                pulse = 0.5 + 0.5 * math.sin(t * 12)
                draw(DrawCommand("circle", cx, cy, radius=s * (0.9 + 0.1 * pulse),
                                 colour=GUARD_GLOW, thickness=3, layer=1))
                draw(DrawCommand("circle", cx, cy, radius=s * 0.6,
                                 colour=_dim(GUARD_GLOW, 0.6), thickness=1, layer=1))

        # Short-lived effects. Colours dim towards black as they expire,
        # because a DrawCommand has no transparency.
        alive = []
        for e in self.fx:
            age = (t - e.t0) / e.duration
            if age > 1.0:
                continue
            alive.append(e)
            k = 1.0 - age
            if e.kind == "slash":
                pts = e.points[int(age * (len(e.points) - 1)):]   # the tail shrinks
                for i in range(1, len(pts)):
                    w = max(1, int(k * 0.22 * e.size * i / len(pts)))
                    colour = SPARK if i == len(pts) - 1 else e.colour
                    draw(DrawCommand("line", *pts[i - 1], *pts[i],
                                     colour=_dim(colour, k), thickness=w, layer=1))
            elif e.kind == "spark":
                for i in range(10):
                    a = i * math.pi / 5 + e.t0
                    r1, r2 = e.size * (0.2 + 0.6 * age), e.size * (0.5 + 0.6 * age)
                    draw(DrawCommand("line", e.x + r1 * math.cos(a), e.y + r1 * math.sin(a),
                                     e.x + r2 * math.cos(a), e.y + r2 * math.sin(a),
                                     colour=_dim(e.colour, k), thickness=max(1, int(4 * k)),
                                     layer=1))
                draw(DrawCommand("circle", e.x, e.y, radius=e.size * (0.3 + 0.7 * age),
                                 colour=_dim(e.colour, k), thickness=2, layer=1))
            elif e.kind == "text":
                scale = max(0.6, e.size / 45)
                draw(DrawCommand("text", e.x - len(e.text) * 10 * scale,
                                 e.y - 0.8 * e.size * age, text=e.text,
                                 colour=_dim(e.colour, 0.4 + 0.6 * k),
                                 thickness=2, layer=2, scale=scale))
            elif e.kind == "ghost":
                c, s = _dim(e.colour, 0.7 * k), e.size
                draw(DrawCommand("circle", e.x, e.y - 0.8 * s, radius=0.35 * s,
                                 colour=c, thickness=1, layer=1))
                draw(DrawCommand("line", e.x - 0.5 * s, e.y, e.x + 0.5 * s, e.y,
                                 colour=c, thickness=1, layer=1))
                draw(DrawCommand("line", e.x, e.y, e.x, e.y + 1.5 * s,
                                 colour=c, thickness=1, layer=1))
        self.fx = alive


# ================================================================ helpers
def _in_pixels(p: Player) -> Player:
    """A copy of the player with keypoints in frame pixels.

    Pose writes keypoints normalised to player.crop (its padded, smoothed
    crop). Measuring in pixels matters for gameplay, not just drawing: the
    crop follows the player, so in crop units a sidestep or a duck barely
    moves the shoulders at all.

    Without a crop the keypoints are taken to be pixels already (the tests,
    the scripted demo).
    """
    box = getattr(p, "crop", None)
    if box is None or not p.keypoints:
        return p
    w, h = box.w, box.h
    return replace(p, keypoints={
        name: Keypoint(box.x0 + kp.x * w, box.y0 + kp.y * h, kp.confidence,
                       kp.vx * w, kp.vy * h)
        for name, kp in p.keypoints.items()})


def _trim(hist: deque, t: float, keep_s: float = 0.6) -> None:
    while hist and t - hist[0][0] > keep_s:
        hist.popleft()


def _velocity(hist: deque):
    """Velocity from the newest sample and one about SPEED_WINDOW_S older.

    Two samples a few frames apart instead of neighbouring frames: keypoint
    jitter of a few pixels would otherwise look like a very fast hand.
    """
    if len(hist) < 2:
        return None
    t1, x1, y1 = hist[-1]
    for t0, x0, y0 in reversed(hist):
        if t1 - t0 >= SPEED_WINDOW_S:
            break
    else:
        return None
    dt = t1 - t0
    if dt > MAX_GAP_S:
        return None
    return (x1 - x0) / dt, (y1 - y0) / dt


def _colour(p: Player) -> tuple[int, int, int]:
    """The player's colour from identity, or a fixed one if none was set."""
    if p.colour != (255, 255, 255):
        return p.colour
    return PALETTE[p.player_id % len(PALETTE)]


def _dim(colour, k: float) -> tuple[int, int, int]:
    return tuple(int(c * k) for c in colour)


def _health_colour(health: int) -> tuple[int, int, int]:
    if health > 50:
        return (80, 200, 80)
    if health > 25:
        return (40, 190, 230)
    return (50, 50, 230)
