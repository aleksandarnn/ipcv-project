# SHARED DATA TYPES - read this before writing any module.
#
# IPCV group project, webcam-based multiplayer game.
# Owner of this file: Aleksandar Nikolov (s2523477), task 5, integration.
#
# Everyone codes against these types. If you need something that is not here,
# tell me and we change them together. Do not invent your own format
# in your own module, because then integration becomes a rewrite.
#
# FOUR RULES WE AGREED ON
#   1. Only main.py opens the webcam. Your module receives a frame.
#   2. Your module NEVER draws on the frame. Return DrawCommands instead,
#      and the renderer decides when and where they appear.
#   3. Your module has a time budget (see BUDGET_MS). Going over it costs
#      us frame rate, which is graded.
#   4. Your module must not crash the game. main.py catches exceptions, but
#      a module that throws every frame gets switched off automatically.
from __future__ import annotations

from dataclasses import dataclass, field

# Milliseconds per frame each module is allowed, at 25 fps (40 ms total).
# These are starting numbers. We measure and renegotiate once modules are real.
BUDGET_MS = {
    "face": 8.0,       # task 1
    "pose": 20.0,      # task 2 — two lite crops @ ≤256px side
    "identity": 3.0,   # task 3
    "control": 3.0,    # task 4
    "render": 6.0,     # task 5
}


@dataclass
class Keypoint:
    """One tracked body point from Task 2 (pose).

    Coordinates are **crop-normalised** relative to that player's bbox from
    Task 3: x,y ∈ [0, 1] across the crop width/height (roughly body-sized).
    Velocities vx,vy are in crop-widths / crop-heights per second.
    Task 4 reads wrist speed via ``speed`` for attack strength.
    """
    x: float
    y: float
    confidence: float = 1.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def ok(self) -> bool:
        # Task 2: anything below this is treated as missing, not as a position.
        return self.confidence >= 0.4

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


@dataclass
class Face:
    """A face box in pixels. angle is the head tilt in degrees, 0 = upright."""
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0
    confidence: float = 1.0


@dataclass
class BBox:
    """Axis-aligned person box in full-frame pixels (Task 3 → Task 2)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(1.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass
class Player:
    """Everything known about one player on this frame.

    player_id is assigned by task 3 and must stay the same person across
    frames, including after they cross, get occluded or leave and come back.
    """
    player_id: int
    face: Face | None = None
    bbox: BBox | None = None                     # Task 3: person crop for pose
    # Task 2: the padded, smoothed box the keypoints are normalised to. It is
    # NOT the same as bbox, so use it to turn keypoints back into pixels:
    #     px = crop.x0 + kp.x * crop.w,  py = crop.y0 + kp.y * crop.h
    crop: BBox | None = None
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    colour: tuple[int, int, int] = (255, 255, 255)   # BGR, for HUD and wands
    health: int = 100
    score: int = 0
    frames_missing: int = 0                          # Task 3: 0 = seen this frame

    @property
    def visible(self) -> bool:
        return self.frames_missing == 0

    def point(self, name: str) -> Keypoint | None:
        """Return a keypoint only if it is trustworthy, otherwise None."""
        kp = self.keypoints.get(name)
        return kp if kp is not None and kp.ok else None


# Keypoint names every module must use. Agreed, not negotiable per person.
KEYPOINTS = (
    # Torso and arms — required for sword attacks and block
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",

    # Legs — required for crouch, jump, and leg attacks
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",

    # Left hand — required later for finger/rune gestures
    "left_thumb_cmc",
    "left_thumb_mcp",
    "left_thumb_ip",
    "left_thumb_tip",
    "left_index_mcp",
    "left_index_pip",
    "left_index_dip",
    "left_index_tip",
    "left_middle_mcp",
    "left_middle_pip",
    "left_middle_dip",
    "left_middle_tip",
    "left_ring_mcp",
    "left_ring_pip",
    "left_ring_dip",
    "left_ring_tip",
    "left_pinky_mcp",
    "left_pinky_pip",
    "left_pinky_dip",
    "left_pinky_tip",

    # Right hand — same structure
    "right_thumb_cmc",
    "right_thumb_mcp",
    "right_thumb_ip",
    "right_thumb_tip",
    "right_index_mcp",
    "right_index_pip",
    "right_index_dip",
    "right_index_tip",
    "right_middle_mcp",
    "right_middle_pip",
    "right_middle_dip",
    "right_middle_tip",
    "right_ring_mcp",
    "right_ring_pip",
    "right_ring_dip",
    "right_ring_tip",
    "right_pinky_mcp",
    "right_pinky_pip",
    "right_pinky_dip",
    "right_pinky_tip",
)


@dataclass
class DrawCommand:
    """A request to draw something. Modules return these; only the renderer draws.

    kind is one of: "circle", "rect", "text", "line".
    layer decides what goes on top: 0 = background, 1 = game, 2 = HUD.
    """
    kind: str
    x: float
    y: float
    x2: float = 0.0
    y2: float = 0.0
    radius: float = 10.0
    text: str = ""
    colour: tuple[int, int, int] = (255, 255, 255)
    thickness: int = 2
    layer: int = 1
    scale: float = 0.6


# The events our game can raise. Task 4 raises them, task 5 turns them into
# something you can see. Spelling matters: an event nobody recognises is
# silently ignored, so agree any new one with me before using it.
#
#   attack      a player swung        strength = how fast the hand moved
#   block       a player has both arms crossed, holding a guard
#   dodge       a player moved out of the way in time
#   damage      an attack landed      player_id = who got hit
#   blocked     an attack was stopped by a guard
#   missed      an attack hit nothing, because the other player dodged
#   ko          a player reached zero health, the round is over
#   round_start a new round is beginning
#   player_lost / player_returned     tracking lost or regained a player
EVENTS = ("attack", "block", "dodge", "damage", "blocked", "missed",
          "ko", "round_start", "player_lost", "player_returned")


@dataclass
class GameEvent:
    """Something happened that the scene should react to.

    player_id is who the event is ABOUT. For "damage" that is the player who
    got hit, not the one who swung.
    """
    kind: str
    player_id: int = -1
    strength: float = 1.0


@dataclass
class GameState:
    """The single source of truth for one frame."""
    frame_index: int = 0
    timestamp: float = 0.0
    players: list[Player] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    draw: list[DrawCommand] = field(default_factory=list)
    phase: str = "playing"        # "waiting", "playing", "paused", "over"
    message: str = ""             # shown big in the middle when not empty

    def player(self, player_id: int) -> Player | None:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None


class Module:
    """What every task module must look like.

    Subclass it, set name, and implement process(). Return nothing; write your
    results into the state. Raise whatever you like, main.py will catch it.
    """
    name = "unnamed"

    def process(self, frame, state: GameState) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Called when the game restarts. Clear any history you keep."""
        pass
