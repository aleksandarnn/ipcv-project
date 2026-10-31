"""Task 2 — body pose estimation and motion tracking.

Contract
--------
**Input** (from Task 3 / identity, via ``state.players``):
  - ``player_id``
  - ``player.bbox`` — ``BBox`` in full-frame pixels (``x0, y0, x1, y1``)

  Pose attaches keypoints onto the Player objects identity already created.
  With identity loaded, ``allow_stubs=False`` (set by main): empty players
  means wait — no fake crops. Stubs (``FIXED_BOXES``) only if identity is off.

**Output** (written onto each ``Player``):
  - ``keypoints[name]`` → ``Keypoint`` with
      - ``x, y``   crop-normalised position in \\[0, 1\\] (relative to bbox)
      - ``vx, vy`` crop-widths / crop-heights per second
      - ``confidence`` (use ``Keypoint.ok``)

  Does **not** own ``frames_missing``, ``bbox``, or ``colour`` when identity
  already set them.

**Debug**: run ``python main.py --debug`` (or set ``Module.debug = True``).
Overlays boxes, skeleton, and velocity arrows. Off by default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from game_types import BBox, Keypoint, Player, player_label

# ---------------------------------------------------------------------------
# Landmark maps / debug skeleton
# ---------------------------------------------------------------------------

# MediaPipe Pose landmark index → shared KEYPOINTS name.
# Body only for now. Finger/hand tips disabled — re-enable when adding
# Hand Landmarker (or pose tip anchors) later.
POSE_TO_KEYPOINT = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    # --- fingers / hand tip anchors (pose model) — disabled for now ---
    # 17: "left_pinky_tip",
    # 18: "right_pinky_tip",
    # 19: "left_index_tip",
    # 20: "right_index_tip",
    # 21: "left_thumb_tip",
    # 22: "right_thumb_tip",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Skeleton edges: (start_idx, end_idx, side) with side in {body, left, right}
BODY_CONNECTIONS = (
    (11, 12, "body"),   # shoulders
    (23, 24, "body"),   # hips
    (11, 23, "body"),   # left shoulder → left hip (torso)
    (12, 24, "body"),   # right shoulder → right hip
    (11, 13, "left"),   # left arm
    (13, 15, "left"),
    (23, 25, "left"),   # left leg
    (25, 27, "left"),
    (27, 29, "left"),
    (29, 31, "left"),
    (12, 14, "right"),  # right arm
    (14, 16, "right"),
    (24, 26, "right"),  # right leg
    (26, 28, "right"),
    (28, 30, "right"),
    (30, 32, "right"),
    # --- wrist → finger tip debug lines — disabled for now ---
    # (15, 17, "left"),
    # (15, 19, "left"),
    # (15, 21, "left"),
    # (16, 18, "right"),
    # (16, 20, "right"),
    # (16, 22, "right"),
)

# BGR skeleton lines (distinct from joint tier colours)
SIDE_DRAW = {
    "body": (0, 255, 0),      # green — torso (facing camera)
    "left": (255, 0, 0),      # blue — person's left
    "right": (0, 0, 255),     # red — person's right
}
# Torso when the model thinks the back faces the camera
SIDE_DRAW_BACK_BODY = (0, 100, 0)     # dark green
# Frames of agreement before flipping facing_back (anti-flicker)
FACING_HYSTERESIS = 4
FACE_LANDMARK_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# ---------------------------------------------------------------------------
# Detection / inference
# ---------------------------------------------------------------------------

VISIBILITY_MIN = 0.5
# # HAND_CONF_MIN = 0.25
# Pose network input (longest crop side). Higher = sharper joints, slower.
INFER_MAX_SIDE = 448
MAX_SPEED = 4000.0  # full-frame px/s clamp inside the tracker

# Crop-normalised velocity (fractions of crop size per second)
VEL_OUTLIER_MAX = 6.0       # above this → zeroed in the returned Keypoint
VEL_ARROW_MIN = 0.40        # draw arrow only if |v| ≥ this
VEL_ARROW_DT = 0.18         # arrow length = velocity × this many seconds
# Velocity is NOT one-frame Δpos (that jitters even when pos looks smooth).
# Use a lookback window + EMA on the resulting velocity.
VEL_LOOKBACK_S = 0.15       # seconds between samples used for Δpos/Δt
VEL_EMA_ALPHA = 0.28        # blend of new lookback vel vs previous vel

# # HAND_TIP_NAMES = frozenset({...})

# ---------------------------------------------------------------------------
# 1. Temporal filter — One-Euro (lower cutoff / beta = smoother, more lag)
# ---------------------------------------------------------------------------

# (min_cutoff_hz, beta) — aggressive smoothing to kill jitter
JOINT_ONE_EURO = {
    "left_wrist": (0.55, 0.22),
    "right_wrist": (0.55, 0.22),
    "left_elbow": (0.45, 0.14),
    "right_elbow": (0.45, 0.14),
    "left_ankle": (0.45, 0.14),
    "right_ankle": (0.45, 0.14),
    "left_knee": (0.40, 0.10),
    "right_knee": (0.40, 0.10),
    "left_shoulder": (0.30, 0.05),
    "right_shoulder": (0.30, 0.05),
    "left_hip": (0.30, 0.05),
    "right_hip": (0.30, 0.05),
    "left_heel": (0.40, 0.10),
    "right_heel": (0.40, 0.10),
    "left_foot_index": (0.40, 0.10),
    "right_foot_index": (0.40, 0.10),
}
DEFAULT_ONE_EURO = (0.40, 0.10)
DERIV_CUTOFF_HZ = 0.5

# ---------------------------------------------------------------------------
# 2. Geometric constraints (relative to torso scale)
# ---------------------------------------------------------------------------

# Bone → (parent, child, min_frac, max_frac) of torso length
# Legs allow wider range — crouch foreshortens thigh/shin in the image.
BONES = (
    ("left_shoulder", "left_elbow", 0.25, 0.95),
    ("left_elbow", "left_wrist", 0.25, 0.95),
    ("right_shoulder", "right_elbow", 0.25, 0.95),
    ("right_elbow", "right_wrist", 0.25, 0.95),
    ("left_hip", "left_knee", 0.20, 1.35),
    ("left_knee", "left_ankle", 0.20, 1.35),
    ("right_hip", "right_knee", 0.20, 1.35),
    ("right_knee", "right_ankle", 0.20, 1.35),
    ("left_shoulder", "right_shoulder", 0.20, 0.90),
    ("left_hip", "right_hip", 0.15, 0.80),
)

# Joint → (proximal, joint, distal, min_deg, max_deg) interior angle
# Knees: allow deep crouch (more acute) and near-straight standing.
JOINT_ANGLES = (
    ("left_shoulder", "left_elbow", "left_wrist", 25.0, 185.0),
    ("right_shoulder", "right_elbow", "right_wrist", 25.0, 185.0),
    ("left_hip", "left_knee", "left_ankle", 20.0, 185.0),
    ("right_hip", "right_knee", "right_ankle", 20.0, 185.0),
)

TORSO_FALLBACK_PX = 120.0

# ---------------------------------------------------------------------------
# Profile (side-view) completion — fill a full skeleton from a thin torso
# ---------------------------------------------------------------------------
# When shoulders nearly collapse to a line, MediaPipe drops most limbs.
# We keep every real detection and synthesise the rest from an anatomical
# template in torso-local coordinates (plus any stored rel_local memory).
PROFILE_SHOULDER_FRAC = 0.42   # shoulder_width / torso_len below this → profile
PROFILE_VIS_MIN = 0.28         # accept weaker landmarks while in / near profile
PROFILE_CONF = 0.58            # confidence for template-filled joints
# Rest-pose offsets in torso frame: origin = shoulder mid, +y → hips, +x → right
# Units = torso length (shoulder mid → hip mid).
ANATOMICAL_LOCAL = {
    "left_shoulder": (-0.38, 0.00),
    "right_shoulder": (0.38, 0.00),
    "left_hip": (-0.28, 1.00),
    "right_hip": (0.28, 1.00),
    "left_elbow": (-0.62, 0.42),
    "right_elbow": (0.62, 0.42),
    "left_wrist": (-0.72, 0.78),
    "right_wrist": (0.72, 0.78),
    "left_knee": (-0.30, 1.55),
    "right_knee": (0.30, 1.55),
    "left_ankle": (-0.28, 2.10),
    "right_ankle": (0.28, 2.10),
    "left_heel": (-0.28, 2.18),
    "right_heel": (0.28, 2.18),
    "left_foot_index": (-0.18, 2.22),
    "right_foot_index": (0.18, 2.22),
}
TORSO_CORE = frozenset({
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
})

# ---------------------------------------------------------------------------
# 3. Occlusion / leave-frame policy
# ---------------------------------------------------------------------------

# Point-level: short constant-velocity bridge, then hold+decay, then drop
PREDICT_FRAMES = 8
HOLD_DECAY = 0.90
# Person-level: after this many full misses, clear the track
MAX_PERSON_MISS = 45
# Re-acquire: need this many consecutive good person detections before
# velocities are trusted again (blocks false attacks on re-entry)
REACQUIRE_FRAMES = 4
# Extra decay when entering / staying in a non-fresh tier (once per frame)
PREDICT_DECAY = 0.92
HOLD_DECAY_EXTRA = 0.85

# ---------------------------------------------------------------------------
# Limb memory (occlusion / teleport) — anchored to shoulder (arms) / hip (legs)
# ---------------------------------------------------------------------------
# Offsets live in the torso orientation, but the origin is the limb root so a
# crouch (hips move) carries the legs with the hips instead of freezing them
# in standing torso space.

RELATIVE_JOINTS = frozenset({
    "left_elbow", "left_wrist",
    "right_elbow", "right_wrist",
    "left_knee", "left_ankle", "left_heel", "left_foot_index",
    "right_knee", "right_ankle", "right_heel", "right_foot_index",
})
JOINT_ANCHOR = {
    "left_elbow": "left_shoulder",
    "left_wrist": "left_shoulder",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_shoulder",
    "left_knee": "left_hip",
    "left_ankle": "left_hip",
    "left_heel": "left_hip",
    "left_foot_index": "left_hip",
    "right_knee": "right_hip",
    "right_ankle": "right_hip",
    "right_heel": "right_hip",
    "right_foot_index": "right_hip",
}
LEG_JOINTS = frozenset({
    "left_knee", "left_ankle", "left_heel", "left_foot_index",
    "right_knee", "right_ankle", "right_heel", "right_foot_index",
})
TORSO_JOINTS = frozenset({
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
})

RELATIVE_MAX_FRAMES = 40
RELATIVE_CONF = 0.70
RELATIVE_DECAY = 0.97
# vs anchor-relative memory
TELEPORT_TORSO_FRAC = 0.50
# vs previous frame position (catches spikes even without memory)
JUMP_FRAC_DEFAULT = 0.35
JUMP_FRAC_TORSO = 0.70          # crouch moves hips/shoulders a lot
JUMP_FRAC_LEG = 0.55
JUMP_FRAC_ARM = 0.40
TELEPORT_OVERRIDE_FRAMES = 4
MEMORY_WARMUP_FRAMES = 5
# If torso scale changes by more than this, drop leg memory (crouch/stand)
SCALE_CHANGE_CLEAR = 0.15

# ---------------------------------------------------------------------------
# 4. Crop / box robustness
# ---------------------------------------------------------------------------

BOX_PAD_FRAC = 0.12          # expand each side by this fraction of box size
BOX_EMA_ALPHA = 0.35         # smooth box edges over time
MIN_CROP_SIDE = 64           # reject / fall back if crop too small
# Pose torso center must stay inside padded box (normalized inset)
IN_BOX_MARGIN_FRAC = 0.02

# Fixed stub boxes while Task 3 is not ready (fractions of frame size).
FIXED_BOXES = (
    (0, 0.02, 0.05, 0.46, 0.95),
    (1, 0.54, 0.05, 0.98, 0.95),
)
BOX_COLOURS = {
    0: (255, 80, 80),
    1: (80, 80, 255),
}

# Debug draw colours by tier
TIER_DRAW = {
    "fresh": (0, 255, 255),      # cyan
    "predicted": (255, 200, 0),  # light blue
    "relative": (255, 0, 255),   # magenta — torso-relative memory
    "held": (0, 165, 255),       # orange
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class OneEuroFilter2D:
    """One-Euro filter for (x, y). Adapts smoothing to speed."""

    def __init__(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._x: float | None = None
        self._y: float | None = None
        self._dx = 0.0
        self._dy = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x = self._y = self._t = None
        self._dx = self._dy = 0.0

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self._x is None or self._t is None:
            self._x, self._y, self._t = x, y, t
            return x, y
        dt = max(t - self._t, 1e-6)
        self._t = t

        a_d = self._alpha(DERIV_CUTOFF_HZ, dt)
        dx = (x - self._x) / dt
        dy = (y - self._y) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        self._dy = a_d * dy + (1.0 - a_d) * self._dy

        speed = (self._dx * self._dx + self._dy * self._dy) ** 0.5
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._y = a * y + (1.0 - a) * self._y
        return self._x, self._y


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Interior angle ABC in degrees."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = (bax * bax + bay * bay) ** 0.5
    nc = (bcx * bcx + bcy * bcy) ** 0.5
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cos_a = (bax * bcx + bay * bcy) / (na * nc)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def _torso_scale(pts: dict[str, tuple[float, float, float]]) -> float:
    """Characteristic body size from shoulder/hip midpoints."""
    def mid(a: str, b: str):
        if a in pts and b in pts:
            return ((pts[a][0] + pts[b][0]) * 0.5, (pts[a][1] + pts[b][1]) * 0.5)
        return None

    sh = mid("left_shoulder", "right_shoulder")
    hp = mid("left_hip", "right_hip")
    if sh is not None and hp is not None:
        d = _dist(sh, hp)
        if d > 1.0:
            return d
    # Fallback: shoulder width
    if "left_shoulder" in pts and "right_shoulder" in pts:
        d = _dist(pts["left_shoulder"][:2], pts["right_shoulder"][:2])
        if d > 1.0:
            return d
    return TORSO_FALLBACK_PX


def _torso_center(pts: dict[str, tuple[float, float, float]]):
    xs, ys = [], []
    for name in (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    ):
        if name in pts:
            xs.append(pts[name][0])
            ys.append(pts[name][1])
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _xy(pts: dict, name: str) -> tuple[float, float] | None:
    v = pts.get(name)
    if v is None:
        return None
    return (v[0], v[1])


def _torso_frame(pts: dict):
    """Orthonormal torso frame: origin at shoulder mid, +x toward right shoulder,
    +y toward hips. Coordinates later stored as (local / scale).

    Returns (origin, ex, ey, scale) or None if shoulders are missing.
    In profile (shoulders collapsed) falls back to a vertical torso axis.
    """
    ls = _xy(pts, "left_shoulder")
    rs = _xy(pts, "right_shoulder")
    lh = _xy(pts, "left_hip")
    rh = _xy(pts, "right_hip")

    shoulders = [p for p in (ls, rs) if p is not None]
    hips = [p for p in (lh, rh) if p is not None]

    if ls is not None and rs is not None:
        origin = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
        ex = (rs[0] - ls[0], rs[1] - ls[1])
        en = (ex[0] * ex[0] + ex[1] * ex[1]) ** 0.5
    elif shoulders:
        origin = shoulders[0]
        ex = (1.0, 0.0)
        en = 0.0
    else:
        return None

    if hips:
        hip_mid = (
            sum(p[0] for p in hips) / len(hips),
            sum(p[1] for p in hips) / len(hips),
        )
    else:
        hip_mid = None

    # Wide enough shoulders → normal frontal frame
    if en >= 3.0 and hip_mid is not None:
        ex = (ex[0] / en, ex[1] / en)
        ey = (-ex[1], ex[0])
        to_hip = (hip_mid[0] - origin[0], hip_mid[1] - origin[1])
        if to_hip[0] * ey[0] + to_hip[1] * ey[1] < 0:
            ey = (-ey[0], -ey[1])
        scale = _dist(origin, hip_mid)
        if scale < 1.0:
            scale = TORSO_FALLBACK_PX
        return origin, ex, ey, scale

    # Profile / collapsed shoulders: +y along torso, +x sideways in the image
    if hip_mid is None:
        return None
    ey = (hip_mid[0] - origin[0], hip_mid[1] - origin[1])
    el = (ey[0] * ey[0] + ey[1] * ey[1]) ** 0.5
    if el < 1.0:
        return None
    ey = (ey[0] / el, ey[1] / el)
    ex = (-ey[1], ey[0])
    # Prefer +x toward image-right for stable L/R colours in profile
    if ex[0] < 0:
        ex = (-ex[0], -ex[1])
    return origin, ex, ey, el


def _is_profile(pts: dict) -> bool:
    """True when the silhouette is basically a torso line (side view)."""
    ls = _xy(pts, "left_shoulder")
    rs = _xy(pts, "right_shoulder")
    lh = _xy(pts, "left_hip")
    rh = _xy(pts, "right_hip")
    if ls is None or rs is None:
        # Single shoulder visible → treat as profile-ish
        return (ls is not None or rs is not None) and (
            lh is not None or rh is not None
        )
    sh_w = _dist(ls, rs)
    if lh is not None and rh is not None:
        hip_mid = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
    elif lh is not None:
        hip_mid = lh
    elif rh is not None:
        hip_mid = rh
    else:
        return sh_w < 20.0
    sh_mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    torso = _dist(sh_mid, hip_mid)
    if torso < 1.0:
        return True
    return (sh_w / torso) < PROFILE_SHOULDER_FRAC


def _to_local(
    p: tuple[float, float],
    origin: tuple[float, float],
    ex: tuple[float, float],
    ey: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    dx, dy = p[0] - origin[0], p[1] - origin[1]
    return (
        (dx * ex[0] + dy * ex[1]) / scale,
        (dx * ey[0] + dy * ey[1]) / scale,
    )


def _from_local(
    local: tuple[float, float],
    origin: tuple[float, float],
    ex: tuple[float, float],
    ey: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    lx, ly = local
    return (
        origin[0] + scale * (lx * ex[0] + ly * ey[0]),
        origin[1] + scale * (lx * ex[1] + ly * ey[1]),
    )


@dataclass
class _PointState:
    """Per-keypoint temporal bookkeeping."""

    x: float = 0.0
    y: float = 0.0
    conf: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    miss: int = 0          # consecutive frames without a fresh measurement
    tier: str = "fresh"    # fresh | predicted | relative | held
    filt: OneEuroFilter2D | None = None
    # Recent smoothed positions for lookback velocity: (t, x, y)
    hist: deque = field(default_factory=lambda: deque(maxlen=24))


@dataclass
class _TrackState:
    """Per-player temporal state (one MediaPipe VIDEO instance each)."""

    landmarker: object
    last_timestamp_ms: int = 0
    frames_missing: int = 0          # person-level miss streak
    good_streak: int = 0             # consecutive person detections (re-acquire)
    acquired: bool = False           # True once REACQUIRE_FRAMES reached
    points: dict[str, _PointState] = field(default_factory=dict)
    prev_time: float | None = None
    # Smoothed crop box in full-frame pixels: (x0, y0, x1, y1)
    box_smooth: tuple[float, float, float, float] | None = None
    last_good_box: tuple[int, int, int, int] | None = None
    # name → (lx, ly) in torso axes, origin = JOINT_ANCHOR, unit = torso_scale
    rel_local: dict[str, tuple[float, float]] = field(default_factory=dict)
    # name → consecutive frames a fresh meas disagreed with rel_local / jump
    rel_disagree: dict[str, int] = field(default_factory=dict)
    jump_disagree: dict[str, int] = field(default_factory=dict)
    # Frames spent acquired (gates when relative memory may be used)
    acquired_frames: int = 0
    mem_scale: float | None = None
    # Back-facing estimate (MediaPipe); hysteresis via facing_back_votes
    facing_back: bool = False
    facing_back_votes: int = 0
    # Side-on to camera — use anatomical template for missing limbs
    profile: bool = False


class Module:
    name = "pose"

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        models = project_root / "models"
        lite = models / "pose_landmarker_lite.task"
        full = models / "pose_landmarker_full.task"
        if lite.exists():
            self._model_path = lite
        elif full.exists():
            self._model_path = full
        else:
            raise FileNotFoundError(
                f"Pose model not found in {models} "
                "(need pose_landmarker_lite.task or pose_landmarker_full.task)"
            )

        self._tracks: dict[int, _TrackState] = {}
        # Overlay boxes / skeleton / velocity arrows (enable with --debug).
        self.debug = False
        # When identity is in the pipeline, main sets this False so we never
        # invent the old left/right stub crops while waiting for detections.
        self.allow_stubs = True

    def _make_landmarker(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _track_for(self, player_id: int) -> _TrackState:
        if player_id not in self._tracks:
            self._tracks[player_id] = _TrackState(
                landmarker=self._make_landmarker()
            )
        return self._tracks[player_id]

    # ------------------------------------------------------------------
    # 4. Boxes: pad, smooth, min size
    # ------------------------------------------------------------------

    def _stub_boxes_px(self, frame) -> list[tuple[int, int, int, int, int]]:
        """Fixed left/right crops when identity has not populated players."""
        h, w = frame.shape[:2]
        out = []
        for player_id, nx0, ny0, nx1, ny1 in FIXED_BOXES:
            out.append((
                player_id,
                int(nx0 * w), int(ny0 * h),
                int(nx1 * w), int(ny1 * h),
            ))
        return out

    @staticmethod
    def _bbox_xyxy(bbox) -> tuple[int, int, int, int] | None:
        """Normalise Task 3 ``BBox`` (or legacy tuple) to int xyxy."""
        if bbox is None:
            return None
        if isinstance(bbox, BBox):
            return int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)
        # Legacy (x1, y1, x2, y2) tuple from older identity builds
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return None

    def _input_boxes_px(self, frame, state) -> list[tuple[int, int, int, int, int]]:
        """Resolve (player_id, x0, y0, x1, y1) from identity, else stubs.

        When identity is loaded (``allow_stubs=False``), an empty player list
        means "nobody detected yet" — return no crops. Never invent the old
        fixed left/right boxes in that case.

        Stubs remain only for solo pose testing with identity disabled.
        """
        out: list[tuple[int, int, int, int, int]] = []
        for player in state.players:
            xyxy = self._bbox_xyxy(player.bbox)
            if xyxy is None:
                continue
            x0, y0, x1, y1 = xyxy
            out.append((player.player_id, x0, y0, x1, y1))
        if out or state.players or not self.allow_stubs:
            return out
        return self._stub_boxes_px(frame)

    def _pad_box(
        self,
        x0: int, y0: int, x1: int, y1: int,
        frame_w: int, frame_h: int,
    ) -> tuple[int, int, int, int]:
        bw, bh = x1 - x0, y1 - y0
        px = int(bw * BOX_PAD_FRAC)
        py = int(bh * BOX_PAD_FRAC)
        return (
            max(0, x0 - px),
            max(0, y0 - py),
            min(frame_w, x1 + px),
            min(frame_h, y1 + py),
        )

    def _smooth_box(
        self,
        track: _TrackState,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = map(float, box)
        if track.box_smooth is None:
            track.box_smooth = (x0, y0, x1, y1)
        else:
            a = BOX_EMA_ALPHA
            sx0, sy0, sx1, sy1 = track.box_smooth
            track.box_smooth = (
                a * x0 + (1 - a) * sx0,
                a * y0 + (1 - a) * sy0,
                a * x1 + (1 - a) * sx1,
                a * y1 + (1 - a) * sy1,
            )
        s = track.box_smooth
        return int(s[0]), int(s[1]), int(s[2]), int(s[3])

    def _iter_crops(self, frame, state):
        """Yield (player_id, crop, offset_xy, box_xyxy) after pad/smooth/min-size."""
        fh, fw = frame.shape[:2]
        for player_id, x0, y0, x1, y1 in self._input_boxes_px(frame, state):
            track = self._track_for(player_id)
            x0, y0, x1, y1 = self._pad_box(x0, y0, x1, y1, fw, fh)
            x0, y0, x1, y1 = self._smooth_box(track, (x0, y0, x1, y1))

            bw, bh = x1 - x0, y1 - y0
            if bw < MIN_CROP_SIDE or bh < MIN_CROP_SIDE:
                if track.last_good_box is not None:
                    x0, y0, x1, y1 = track.last_good_box
                else:
                    continue

            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            track.last_good_box = (x0, y0, x1, y1)
            yield player_id, crop, (x0, y0), (x0, y0, x1, y1)

    def process(self, frame, state) -> None:
        """Run pose for each input crop; write crop-normalised keypoints."""
        crops = list(self._iter_crops(frame, state))

        for player_id, crop, offset, box in crops:
            # Internal estimate stays in full-frame pixels for filtering.
            full_kps = self._estimate_player(
                player_id, crop, offset, box, state.timestamp
            )
            keypoints = self._to_crop_normalised(full_kps, box)

            player = self._ensure_player(state, player_id)
            player.keypoints = keypoints
            # The exact box the keypoints are normalised to (padded and
            # smoothed, so not identity's bbox). Task 4 needs it for pixels.
            player.crop = BBox(
                x0=float(box[0]), y0=float(box[1]),
                x1=float(box[2]), y1=float(box[3]),
            )
            # Identity owns bbox / colour / frames_missing. Only fill gaps
            # when running without Task 3 (stub-box path created the player).
            if player.bbox is None:
                player.bbox = BBox(
                    x0=float(box[0]), y0=float(box[1]),
                    x1=float(box[2]), y1=float(box[3]),
                )
            if player.colour == (255, 255, 255):
                player.colour = BOX_COLOURS.get(player_id, player.colour)

            if self.debug and keypoints:
                track = self._tracks[player_id]
                self._draw_debug_keypoints(frame, keypoints, track, box)

        if self.debug:
            self._draw_debug_boxes(frame, crops)

    @staticmethod
    def _to_crop_normalised(
        full_kps: dict[str, Keypoint],
        box: tuple[int, int, int, int],
    ) -> dict[str, Keypoint]:
        """Full-frame px → crop-relative \\[0, 1\\] position / velocity.

        Velocities above ``VEL_OUTLIER_MAX`` (crop units / s) are zeroed.
        """
        x0, y0, x1, y1 = box
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        out: dict[str, Keypoint] = {}
        for name, kp in full_kps.items():
            vx, vy = kp.vx / w, kp.vy / h
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > VEL_OUTLIER_MAX:
                vx = vy = 0.0
            out[name] = Keypoint(
                x=(kp.x - x0) / w,
                y=(kp.y - y0) / h,
                confidence=kp.confidence,
                vx=vx,
                vy=vy,
            )
        return out

    def _prepare_infer_image(self, crop):
        h, w = crop.shape[:2]
        side = max(h, w)
        if side > INFER_MAX_SIDE:
            scale = INFER_MAX_SIDE / side
            small = cv2.resize(
                crop,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = crop
        sh, sw = small.shape[:2]
        scale_x = w / sw
        scale_y = h / sh
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return rgb, scale_x, scale_y, sw, sh

    @staticmethod
    def _landmark_confidence(lm, name: str) -> float:
        """Confidence for a landmark. Body: visibility only (fingers disabled)."""
        # --- fingers (disabled): prefer max(visibility, presence) for tips ---
        # visibility = float(getattr(lm, "visibility", 0.0) or 0.0)
        # presence = float(getattr(lm, "presence", 0.0) or 0.0)
        # if name in HAND_TIP_NAMES:
        #     return max(visibility, presence)
        # return visibility
        return float(getattr(lm, "visibility", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # 2. Geometric rejection
    # ------------------------------------------------------------------

    def _reject_geometry(
        self,
        raw: dict[str, tuple[float, float, float]],
    ) -> dict[str, tuple[float, float, float]]:
        """Drop child joints that break bone-length or angle priors."""
        if len(raw) < 2:
            return raw
        scale = _torso_scale(raw)
        pts = {n: (v[0], v[1]) for n, v in raw.items()}
        rejected: set[str] = set()

        for parent, child, lo, hi in BONES:
            if parent in rejected or child not in pts or parent not in pts:
                continue
            d = _dist(pts[parent], pts[child]) / scale
            if d < lo or d > hi:
                rejected.add(child)

        for prox, joint, dist, amin, amax in JOINT_ANGLES:
            if joint in rejected:
                continue
            if prox not in pts or joint not in pts or dist not in pts:
                continue
            if prox in rejected or dist in rejected:
                continue
            ang = _angle_deg(pts[prox], pts[joint], pts[dist])
            if ang < amin or ang > amax:
                rejected.add(dist)

        return {n: v for n, v in raw.items() if n not in rejected}

    # ------------------------------------------------------------------
    # 4. Person-in-box check
    # ------------------------------------------------------------------

    def _pose_in_box(
        self,
        raw: dict[str, tuple[float, float, float]],
        box: tuple[int, int, int, int],
    ) -> bool:
        center = _torso_center(raw)
        if center is None:
            return False
        x0, y0, x1, y1 = box
        mx = (x1 - x0) * IN_BOX_MARGIN_FRAC
        my = (y1 - y0) * IN_BOX_MARGIN_FRAC
        return (x0 + mx) <= center[0] <= (x1 - mx) and \
               (y0 + my) <= center[1] <= (y1 - my)

    # ------------------------------------------------------------------
    # Main per-player estimate
    # ------------------------------------------------------------------

    def _estimate_player(
        self,
        player_id: int,
        crop,
        offset: tuple[int, int],
        box: tuple[int, int, int, int],
        timestamp: float,
    ) -> dict[str, Keypoint]:
        track = self._track_for(player_id)
        ox, oy = offset

        rgb, scale_x, scale_y, sw, sh = self._prepare_infer_image(crop)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(
            track.last_timestamp_ms + 1,
            int(timestamp * 1000),
        )
        track.last_timestamp_ms = timestamp_ms

        result = track.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected = bool(result.pose_landmarks)

        raw: dict[str, tuple[float, float, float]] = {}
        if detected:
            landmarks = result.pose_landmarks[0]
            self._update_facing(track, landmarks)
            # Soften visibility for torso core so profile silhouettes survive
            for idx, name in POSE_TO_KEYPOINT.items():
                lm = landmarks[idx]
                conf = self._landmark_confidence(lm, name)
                min_conf = (
                    PROFILE_VIS_MIN if name in TORSO_CORE else VISIBILITY_MIN
                )
                if conf < min_conf:
                    continue
                x = lm.x * sw * scale_x + ox
                y = lm.y * sh * scale_y + oy
                raw[name] = (x, y, conf)

            # Second pass: if already looking like a profile line, keep weaker limbs
            if _is_profile(self._frame_pts(track, raw)):
                for idx, name in POSE_TO_KEYPOINT.items():
                    if name in raw:
                        continue
                    lm = landmarks[idx]
                    conf = self._landmark_confidence(lm, name)
                    if conf < PROFILE_VIS_MIN:
                        continue
                    x = lm.x * sw * scale_x + ox
                    y = lm.y * sh * scale_y + oy
                    raw[name] = (x, y, conf)

            track.profile = _is_profile(self._frame_pts(track, raw))

            # Geometry rejects shoulder width in profile — skip when side-on
            if track.acquired and not track.profile:
                raw = self._reject_geometry(raw)

            # Pose found but torso outside this player's box → treat as miss
            if not raw or not self._pose_in_box(raw, box):
                detected = False
                raw = {}
                track.profile = False
            else:
                # Expand a single shoulder/hip into a pair so the frame exists
                raw = self._seed_torso_pair(raw)
                dt_hint = None
                if track.prev_time is not None:
                    dt_hint = timestamp - track.prev_time
                    if dt_hint <= 1e-6:
                        dt_hint = None
                raw = self._reject_frame_jumps(track, raw, dt_hint)
                if self._memory_ready(track):
                    if not track.profile:
                        self._maybe_clear_leg_memory(track, raw)
                        raw = self._reject_teleports(track, raw)
                    self._update_rel_memory(track, raw)
        else:
            track.facing_back_votes = 0
            track.profile = False

        was_acquired = track.acquired
        if detected:
            track.frames_missing = 0
            track.good_streak += 1
            if track.good_streak >= REACQUIRE_FRAMES:
                track.acquired = True
        else:
            track.frames_missing += 1
            track.good_streak = 0
            if track.frames_missing >= REACQUIRE_FRAMES:
                track.acquired = False

        if track.acquired:
            if not was_acquired:
                self._on_acquired(track)
            track.acquired_frames += 1
            if self._memory_ready(track) and raw:
                self._update_rel_memory(track, raw)
        else:
            if was_acquired:
                track.rel_local.clear()
                track.rel_disagree.clear()
                track.jump_disagree.clear()
                track.mem_scale = None
            track.acquired_frames = 0

        if track.frames_missing > MAX_PERSON_MISS:
            self._clear_points(track)
            return {}

        dt = None
        if track.prev_time is not None:
            dt = timestamp - track.prev_time
            if dt <= 1e-6:
                dt = None

        relative = (
            self._relative_completions(track, raw)
            if self._memory_ready(track) else {}
        )
        # Profile (or sparse pose): fill remaining joints from anatomy template
        if track.acquired and raw:
            template = self._template_completions(track, raw, relative)
            for name, val in template.items():
                relative.setdefault(name, val)

        keypoints = self._update_points(track, raw, relative, timestamp, dt)
        track.prev_time = timestamp
        return keypoints

    def _memory_ready(self, track: _TrackState) -> bool:
        return track.acquired and track.acquired_frames >= MEMORY_WARMUP_FRAMES

    def _on_acquired(self, track: _TrackState) -> None:
        """Reset sticky state so a bad warmup pose cannot lock forever."""
        track.rel_local.clear()
        track.rel_disagree.clear()
        track.jump_disagree.clear()
        track.mem_scale = None
        track.acquired_frames = 0
        for ps in track.points.values():
            if ps.filt is not None:
                ps.filt.reset()

    def _clear_points(self, track: _TrackState) -> None:
        track.points.clear()
        track.rel_local.clear()
        track.rel_disagree.clear()
        track.jump_disagree.clear()
        track.mem_scale = None
        track.prev_time = None
        track.good_streak = 0
        track.acquired = False
        track.acquired_frames = 0
        track.facing_back = False
        track.facing_back_votes = 0
        track.profile = False

    # ------------------------------------------------------------------
    # Profile / anatomical template
    # ------------------------------------------------------------------

    def _seed_torso_pair(
        self, raw: dict[str, tuple[float, float, float]]
    ) -> dict[str, tuple[float, float, float]]:
        """If only one shoulder or hip is seen, invent its partner nearby.

        Gives ``_torso_frame`` enough anchors to place a full template.
        """
        out = dict(raw)
        pts = out
        frame_guess = _torso_frame(pts)
        # Need at least one shoulder + one hip for a line
        has_sh = "left_shoulder" in out or "right_shoulder" in out
        has_hp = "left_hip" in out or "right_hip" in out
        if not (has_sh and has_hp):
            return out

        # Estimate lateral unit from existing pair or default image-x
        if frame_guess is not None:
            _o, ex, _ey, scale = frame_guess
        else:
            # Build a crude down-axis
            sh = _xy(pts, "left_shoulder") or _xy(pts, "right_shoulder")
            hp = _xy(pts, "left_hip") or _xy(pts, "right_hip")
            if sh is None or hp is None:
                return out
            ey = (hp[0] - sh[0], hp[1] - sh[1])
            el = (ey[0] * ey[0] + ey[1] * ey[1]) ** 0.5
            if el < 1.0:
                return out
            ey = (ey[0] / el, ey[1] / el)
            ex = (-ey[1], ey[0])
            if ex[0] < 0:
                ex = (-ex[0], -ex[1])
            scale = el

        half_sh = 0.38 * scale
        half_hp = 0.28 * scale
        conf = PROFILE_CONF

        if "left_shoulder" in out and "right_shoulder" not in out:
            x, y, _ = out["left_shoulder"]
            out["right_shoulder"] = (
                x + 2 * half_sh * ex[0], y + 2 * half_sh * ex[1], conf,
            )
        elif "right_shoulder" in out and "left_shoulder" not in out:
            x, y, _ = out["right_shoulder"]
            out["left_shoulder"] = (
                x - 2 * half_sh * ex[0], y - 2 * half_sh * ex[1], conf,
            )

        if "left_hip" in out and "right_hip" not in out:
            x, y, _ = out["left_hip"]
            out["right_hip"] = (
                x + 2 * half_hp * ex[0], y + 2 * half_hp * ex[1], conf,
            )
        elif "right_hip" in out and "left_hip" not in out:
            x, y, _ = out["right_hip"]
            out["left_hip"] = (
                x - 2 * half_hp * ex[0], y - 2 * half_hp * ex[1], conf,
            )
        return out

    def _template_completions(
        self,
        track: _TrackState,
        raw: dict,
        already: dict,
    ) -> dict[str, tuple[float, float, float]]:
        """Place missing joints from the anatomical rest-pose template.

        Real detections in ``raw`` always win. ``already`` (relative memory)
        wins over the template. Always available once a torso frame exists;
        especially important in profile.
        """
        pts = self._frame_pts(track, raw)
        # Need a usable torso axis
        if not _is_profile(pts) and not track.profile:
            # Frontal: only fill if we are very sparse (< 6 live joints)
            if sum(1 for n in TORSO_CORE if n in raw) < 2:
                return {}
            if len(raw) >= 8:
                return {}

        frame = _torso_frame(pts)
        if frame is None:
            return {}
        origin, ex, ey, scale = frame

        out: dict[str, tuple[float, float, float]] = {}
        for name, local in ANATOMICAL_LOCAL.items():
            if name in raw or name in already:
                continue
            # Prefer stored limb memory over the generic rest-pose when possible
            use_local = track.rel_local.get(name, local)
            x, y = _from_local(use_local, origin, ex, ey, scale)
            out[name] = (x, y, PROFILE_CONF)
        return out

    # ------------------------------------------------------------------
    # Facing (front vs back)
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_facing_back(landmarks) -> bool:
        """Heuristic: low face visibility and/or torso normal pointing away.

        MediaPipe keeps anatomical L/R; when the back faces the camera the face
        landmarks usually disappear and the shoulder→hip cross product flips.
        """
        face_vis = [
            float(getattr(landmarks[i], "visibility", 0.0) or 0.0)
            for i in FACE_LANDMARK_IDS
        ]
        mean_face = sum(face_vis) / max(len(face_vis), 1)
        nose = landmarks[0]
        ls, rs = landmarks[11], landmarks[12]
        lh, rh = landmarks[23], landmarks[24]

        # Nose closer to camera than shoulders → usually facing toward camera
        shoulder_z = 0.5 * (float(ls.z) + float(rs.z))
        nose_closer = float(nose.z) < shoulder_z - 0.02

        # 2D cross: (left→right shoulder) × (shoulder mid→hip mid); y grows down.
        # Facing camera with L on the left typically gives cross > 0.
        mid_sx = 0.5 * (ls.x + rs.x)
        mid_sy = 0.5 * (ls.y + rs.y)
        mid_hx = 0.5 * (lh.x + rh.x)
        mid_hy = 0.5 * (lh.y + rh.y)
        ax, ay = rs.x - ls.x, rs.y - ls.y
        bx, by = mid_hx - mid_sx, mid_hy - mid_sy
        cross = ax * by - ay * bx
        shoulders_rear = cross < 0.0

        if mean_face < 0.35:
            return True
        if mean_face > 0.55 and nose_closer and not shoulders_rear:
            return False
        if shoulders_rear and mean_face < 0.55:
            return True
        return mean_face < 0.45 and not nose_closer

    def _update_facing(self, track: _TrackState, landmarks) -> None:
        guess = self._raw_facing_back(landmarks)
        if guess == track.facing_back:
            track.facing_back_votes = 0
            return
        track.facing_back_votes += 1
        if track.facing_back_votes >= FACING_HYSTERESIS:
            track.facing_back = guess
            track.facing_back_votes = 0

    # ------------------------------------------------------------------
    # Anchor-relative limb memory + jump clamps
    # ------------------------------------------------------------------

    def _frame_pts(self, track: _TrackState, raw: dict) -> dict:
        """Best available positions for building a torso frame (raw overrides)."""
        pts = {}
        for name, ps in track.points.items():
            pts[name] = (ps.x, ps.y, ps.conf)
        pts.update(raw)
        return pts

    @staticmethod
    def _jump_frac(name: str) -> float:
        if name in TORSO_JOINTS:
            return JUMP_FRAC_TORSO
        if name in LEG_JOINTS:
            return JUMP_FRAC_LEG
        if name in RELATIVE_JOINTS:
            return JUMP_FRAC_ARM
        return JUMP_FRAC_DEFAULT

    def _reject_frame_jumps(
        self,
        track: _TrackState,
        raw: dict,
        dt: float | None,
    ) -> dict:
        """Drop single-frame position spikes vs the last published point."""
        pts = self._frame_pts(track, raw)
        frame = _torso_frame(pts)
        scale = frame[3] if frame is not None else TORSO_FALLBACK_PX

        kept = {}
        for name, val in raw.items():
            prev = track.points.get(name)
            if prev is None or prev.miss > 2:
                kept[name] = val
                track.jump_disagree[name] = 0
                continue

            budget = self._jump_frac(name) * scale
            if dt is not None and track.acquired:
                speed = (prev.vx * prev.vx + prev.vy * prev.vy) ** 0.5
                budget = max(budget, 2.2 * speed * dt + 0.12 * scale)

            if _dist((val[0], val[1]), (prev.x, prev.y)) <= budget:
                kept[name] = val
                track.jump_disagree[name] = 0
                continue

            n = track.jump_disagree.get(name, 0) + 1
            track.jump_disagree[name] = n
            if n >= TELEPORT_OVERRIDE_FRAMES:
                kept[name] = val
                track.jump_disagree[name] = 0
            # else: reject spike this frame
        return kept

    def _maybe_clear_leg_memory(self, track: _TrackState, raw: dict) -> None:
        """Crouch/stand changes torso scale — standing leg memory becomes wrong."""
        pts = self._frame_pts(track, raw)
        frame = _torso_frame(pts)
        if frame is None:
            return
        scale = frame[3]
        if track.mem_scale is None:
            track.mem_scale = scale
            return
        if track.mem_scale < 1.0:
            track.mem_scale = scale
            return
        change = abs(scale - track.mem_scale) / track.mem_scale
        if change >= SCALE_CHANGE_CLEAR:
            for name in LEG_JOINTS:
                track.rel_local.pop(name, None)
                track.rel_disagree.pop(name, None)
            track.mem_scale = scale

    def _anchor_frame(self, pts: dict, joint: str):
        """Torso axes + origin at the limb root (shoulder or hip)."""
        frame = _torso_frame(pts)
        if frame is None:
            return None
        _mid, ex, ey, scale = frame
        anchor_name = JOINT_ANCHOR.get(joint)
        if anchor_name is None:
            return None
        anchor = _xy(pts, anchor_name)
        if anchor is None:
            return None
        return anchor, ex, ey, scale

    def _update_rel_memory(self, track: _TrackState, raw: dict) -> None:
        """Store anchor-local offsets for every fresh relative joint."""
        if not self._memory_ready(track):
            return
        need = (
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        )
        pts = self._frame_pts(track, raw)
        if any(n not in pts for n in need):
            return
        frame = _torso_frame(pts)
        if frame is None:
            return
        track.mem_scale = frame[3]

        for name in RELATIVE_JOINTS:
            if name not in raw:
                continue
            af = self._anchor_frame(pts, name)
            if af is None:
                continue
            origin, ex, ey, scale = af
            x, y, _ = raw[name]
            track.rel_local[name] = _to_local((x, y), origin, ex, ey, scale)
            track.rel_disagree[name] = 0

    def _reject_teleports(self, track: _TrackState, raw: dict) -> dict:
        """Drop limb samples far from anchor-relative memory (with override)."""
        if not track.rel_local or not self._memory_ready(track):
            return raw
        pts = self._frame_pts(track, raw)
        kept = {}
        for name, val in raw.items():
            if name not in RELATIVE_JOINTS or name not in track.rel_local:
                kept[name] = val
                continue
            af = self._anchor_frame(pts, name)
            if af is None:
                kept[name] = val
                continue
            origin, ex, ey, scale = af
            expected = _from_local(track.rel_local[name], origin, ex, ey, scale)
            thresh = TELEPORT_TORSO_FRAC * scale
            # Legs: looser while crouching (scale already changing)
            if name in LEG_JOINTS:
                thresh *= 1.25
            if _dist((val[0], val[1]), expected) <= thresh:
                track.rel_disagree[name] = 0
                kept[name] = val
                continue
            n = track.rel_disagree.get(name, 0) + 1
            track.rel_disagree[name] = n
            if n >= TELEPORT_OVERRIDE_FRAMES:
                kept[name] = val
                track.rel_disagree[name] = 0
        return kept

    def _relative_completions(
        self,
        track: _TrackState,
        raw: dict,
    ) -> dict[str, tuple[float, float, float]]:
        """Place missing limb joints from last anchor-relative pose."""
        if not track.rel_local or not self._memory_ready(track):
            return {}
        pts = self._frame_pts(track, raw)
        if any(
            n not in pts
            for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ):
            return {}

        out: dict[str, tuple[float, float, float]] = {}
        for name, local in track.rel_local.items():
            if name in raw:
                continue
            # Need a live anchor so crouch carries the limb correctly
            anchor = JOINT_ANCHOR.get(name)
            if anchor is None or anchor not in pts:
                continue
            af = self._anchor_frame(pts, name)
            if af is None:
                continue
            origin, ex, ey, scale = af
            prev = track.points.get(name)
            miss = 1 if prev is None else prev.miss + 1
            if miss > RELATIVE_MAX_FRAMES:
                continue
            x, y = _from_local(local, origin, ex, ey, scale)
            conf = RELATIVE_CONF * (RELATIVE_DECAY ** max(miss - 1, 0))
            if conf < 0.4:
                continue
            out[name] = (x, y, conf)
        return out

    def _filter_for(self, track: _TrackState, name: str) -> OneEuroFilter2D:
        ps = track.points.get(name)
        if ps is not None and ps.filt is not None:
            return ps.filt
        min_c, beta = JOINT_ONE_EURO.get(name, DEFAULT_ONE_EURO)
        return OneEuroFilter2D(min_c, beta)

    # ------------------------------------------------------------------
    # 1 + 3. One-Euro, relative / predict / hold, re-acquire velocity gate
    # ------------------------------------------------------------------

    @staticmethod
    def _velocity_lookback(
        prev: _PointState | None,
        sx: float,
        sy: float,
        timestamp: float,
        acquired: bool,
        max_speed: float = MAX_SPEED,
    ) -> tuple[float, float, deque]:
        """Smooth velocity from lookback Δpos/Δt + EMA (not one-frame diff)."""
        hist: deque = prev.hist if prev is not None else deque(maxlen=24)
        if hist.maxlen != 24:
            hist = deque(hist, maxlen=24)
        hist.append((timestamp, sx, sy))

        if not acquired or len(hist) < 2:
            return 0.0, 0.0, hist

        target_t = timestamp - VEL_LOOKBACK_S
        old_t, old_x, old_y = hist[0]
        for t, x, y in hist:
            if t <= target_t:
                old_t, old_x, old_y = t, x, y
            else:
                break

        dt = timestamp - old_t
        if dt < VEL_LOOKBACK_S * 0.45:
            # Not enough history yet — keep previous smoothed vel
            if prev is not None:
                return prev.vx, prev.vy, hist
            return 0.0, 0.0, hist

        vx = (sx - old_x) / dt
        vy = (sy - old_y) / dt
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > max_speed:
            s = max_speed / speed
            vx *= s
            vy *= s

        if prev is not None:
            a = VEL_EMA_ALPHA
            vx = a * vx + (1.0 - a) * prev.vx
            vy = a * vy + (1.0 - a) * prev.vy
        return vx, vy, hist

    def _update_points(
        self,
        track: _TrackState,
        raw: dict[str, tuple[float, float, float]],
        relative: dict[str, tuple[float, float, float]],
        timestamp: float,
        dt: float | None,
    ) -> dict[str, Keypoint]:
        names = set(track.points) | set(raw) | set(relative)
        out: dict[str, Keypoint] = {}

        for name in names:
            meas = raw.get(name)
            rel = relative.get(name)
            prev = track.points.get(name)

            if meas is not None:
                x, y, conf = meas
                filt = self._filter_for(track, name)
                sx, sy = filt(x, y, timestamp)

                vx, vy, hist = self._velocity_lookback(
                    prev, sx, sy, timestamp, track.acquired
                )

                track.points[name] = _PointState(
                    x=sx, y=sy, conf=conf, vx=vx, vy=vy,
                    miss=0, tier="fresh", filt=filt, hist=hist,
                )
                out[name] = Keypoint(x=sx, y=sy, confidence=conf, vx=vx, vy=vy)
                continue

            # Prefer torso-relative placement over pixel CV / hold
            if rel is not None:
                x, y, conf = rel
                miss = 1 if prev is None else prev.miss + 1
                filt = self._filter_for(track, name)
                sx, sy = filt(x, y, timestamp)
                vx, vy, hist = self._velocity_lookback(
                    prev, sx, sy, timestamp, track.acquired,
                    max_speed=MAX_SPEED * 0.35,
                )

                track.points[name] = _PointState(
                    x=sx, y=sy, conf=conf, vx=vx, vy=vy,
                    miss=miss, tier="relative", filt=filt, hist=hist,
                )
                out[name] = Keypoint(x=sx, y=sy, confidence=conf, vx=vx, vy=vy)
                continue

            # --- no meas / no relative: CV predict → hold → drop ---
            if prev is None:
                continue

            miss = prev.miss + 1
            if miss <= PREDICT_FRAMES and dt is not None and track.acquired:
                px = prev.x + prev.vx * dt
                py = prev.y + prev.vy * dt
                conf = prev.conf * PREDICT_DECAY
                tier = "predicted"
                vx, vy = prev.vx, prev.vy
            else:
                px, py = prev.x, prev.y
                conf = prev.conf * HOLD_DECAY * HOLD_DECAY_EXTRA
                tier = "held"
                vx = vy = 0.0

            if conf < 0.4:
                del track.points[name]
                continue

            filt = prev.filt
            hist = prev.hist
            hist.append((timestamp, px, py))
            if filt is not None and tier == "predicted":
                filt._x, filt._y, filt._t = px, py, timestamp

            track.points[name] = _PointState(
                x=px, y=py, conf=conf, vx=vx, vy=vy,
                miss=miss, tier=tier, filt=filt, hist=hist,
            )
            out[name] = Keypoint(x=px, y=py, confidence=conf, vx=vx, vy=vy)

        return out

    def _ensure_player(self, state, player_id: int) -> Player:
        player = state.player(player_id)
        if player is not None:
            return player
        player = Player(player_id=player_id)
        state.players.append(player)
        return player

    def reset(self) -> None:
        for track in self._tracks.values():
            try:
                track.landmarker.close()
            except Exception:  # noqa: BLE001
                pass
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Debug draw
    # ------------------------------------------------------------------

    def _draw_debug_boxes(self, frame, crops) -> None:
        for player_id, _crop, _off, box in crops:
            x0, y0, x1, y1 = box
            colour = BOX_COLOURS.get(player_id, (255, 255, 255))
            cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
            track = self._tracks.get(player_id)
            tag = player_label(player_id)
            if track is not None:
                if not track.acquired and track.good_streak > 0:
                    tag += f" lock {track.good_streak}/{REACQUIRE_FRAMES}"
                elif track.frames_missing:
                    tag += f" miss {track.frames_missing}"
                if track.facing_back:
                    tag += " BACK"
                if track.profile:
                    tag += " PROFILE"
            cv2.putText(
                frame, tag, (x0 + 8, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
            )

    def _side_colour(self, side: str, facing_back: bool) -> tuple[int, int, int]:
        """Torso turns purple when back-facing; L/R limb colours swap."""
        if side == "body":
            return SIDE_DRAW_BACK_BODY if facing_back else SIDE_DRAW["body"]
        if facing_back:
            if side == "left":
                return SIDE_DRAW["right"]
            if side == "right":
                return SIDE_DRAW["left"]
        return SIDE_DRAW.get(side, (0, 255, 0))

    def _draw_debug_keypoints(
        self,
        frame,
        keypoints: dict[str, Keypoint],
        track: _TrackState,
        box: tuple[int, int, int, int],
    ) -> None:
        """Draw crop-normalised keypoints back onto the full frame."""
        x0, y0, x1, y1 = box
        bw = max(1.0, float(x1 - x0))
        bh = max(1.0, float(y1 - y0))

        def to_px(kp: Keypoint) -> tuple[int, int]:
            return (int(x0 + kp.x * bw), int(y0 + kp.y * bh))

        index_of = {name: idx for idx, name in POSE_TO_KEYPOINT.items()}
        by_index: dict[int, tuple[int, int, float]] = {}
        for name, kp in keypoints.items():
            idx = index_of.get(name)
            if idx is None:
                continue
            px, py = to_px(kp)
            by_index[idx] = (px, py, kp.confidence)

        for a, b, side in BODY_CONNECTIONS:
            if a not in by_index or b not in by_index:
                continue
            if by_index[a][2] < 0.4 or by_index[b][2] < 0.4:
                continue
            cv2.line(
                frame, by_index[a][:2], by_index[b][:2],
                self._side_colour(side, track.facing_back), 2, cv2.LINE_AA,
            )

        for name, kp in keypoints.items():
            if not kp.ok:
                continue
            tier = "fresh"
            ps = track.points.get(name)
            if ps is not None:
                tier = ps.tier
            colour = TIER_DRAW.get(tier, (0, 255, 255))
            px, py = to_px(kp)
            cv2.circle(frame, (px, py), 8, colour, -1, cv2.LINE_AA)

            speed = kp.speed
            if speed < VEL_ARROW_MIN:
                continue
            # Arrow tip: current pos + velocity × VEL_ARROW_DT (in pixels)
            tip = (
                int(px + kp.vx * bw * VEL_ARROW_DT),
                int(py + kp.vy * bh * VEL_ARROW_DT),
            )
            cv2.arrowedLine(
                frame, (px, py), tip, colour, 2, cv2.LINE_AA, tipLength=0.35,
            )
