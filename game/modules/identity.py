"""
Task 3 — Multiplayer Detection and Identity Tracking
Owner: Akash

Pipeline (runs first, before face and pose):
  raw frame
    → person detection (YOLO / MediaPipe / MOG2, every DETECT_EVERY frames)
    → Kalman prediction (every frame, fills gaps between detections)
    → Stage 1: IoU-based Hungarian assignment  (normal movement)
    → Stage 2: HSV histogram fingerprint match (crossing / occlusion)
    → re-entry matching (player left frame and came back)
    → write state.players with stable player_id + bbox
    → fire player_lost / player_returned events

The histogram is the "DNA" of each player — a compact colour signature
computed from their bounding box crop. It updates slowly so clothing colour
stays stable across the round, and is used whenever pure position-based
matching is ambiguous.

Design references: SORT (Bewley 2016), HopTrack (Li et al. 2024).
"""
from __future__ import annotations

import numpy as np
import cv2

from game_types import DrawCommand, GameEvent, GameState, Module as Base, Player

# ── Detector selection ───────────────────────────────────────────────────────
# Change this to compare detection backends:
#
#   "yolo"       YOLOv8-nano.   Best accuracy. ~17ms/call → ~6ms avg (DETECT_EVERY=3)
#   "mediapipe"  BlazePose lite (Tasks API, 2-person). ~15ms/call → ~5ms avg
#   "mog2"       Background subtraction. ~2ms every frame. Best for fixed camera.
#
# For the 5ms budget:
#   mog2         runs every frame at ~2ms  — fastest, but needs static background
#   mediapipe    DETECT_EVERY=3 → ~5ms avg — good balance
#   yolo         DETECT_EVERY=4 → ~4.5ms avg — reliable, slightly over on peak
#
# Note: OpenCV HOG people detector was removed in OpenCV 5. Not available here.
#
DETECTOR_MODE  = "yolo"   # ← default; override with:  python main.py --detector mog2

# ── Player colours (BGR) ────────────────────────────────────────────────────
COLOURS = {0: (60, 180, 255), 1: (60, 60, 220)}   # P0 = amber, P1 = red

# ── Tuning constants ────────────────────────────────────────────────────────
MAX_LOST       = 25    # frames before a track is fully evicted from state
LOST_HIDE      = 8     # frames before the bounding box stops being drawn
DETECT_EVERY   = 3     # run the detector every N frames; Kalman-predict otherwise
IOU_THRESH     = 0.25  # minimum IoU to accept a Stage-1 match
HIST_THRESH    = 0.35  # minimum histogram correlation to accept a Stage-2 match
HIST_ALPHA     = 0.05  # histogram EMA learning rate (low = stable fingerprint)


# ══════════════════════════════════════════════════════════ Kalman tracker ══

class _KalmanTrack:
    """
    One tracked person.

    State vector: [cx, cy, w, h, vx, vy]
    Constant-velocity model. Observation: [cx, cy, w, h].
    """

    def __init__(self, bbox: tuple[int, int, int, int], player_id: int) -> None:
        self.player_id = player_id
        self.lost      = 0      # frames since last matched detection
        self.hits      = 1      # total matched detections
        self._prev_lost = 0     # lost count at end of previous frame (for events)

        cx, cy, w, h = _bbox_to_cwh(bbox)

        # ── state and covariance ────────────────────────────────────────────
        self._x = np.array([cx, cy, w, h, 0., 0.], dtype=float)
        self._P = np.diag([10., 10., 10., 10., 1000., 1000.])

        # transition (constant velocity: cx += vx, cy += vy each frame)
        self._F      = np.eye(6); self._F[0, 4] = 1.; self._F[1, 5] = 1.
        self._H      = np.eye(4, 6)           # observe first 4 state dims
        self._Q      = np.diag([1., 1., 1., 1., 0.1, 0.1])   # process noise
        self._R      = np.diag([10., 10., 20., 20.])          # measurement noise

        # ── histogram fingerprint (set after first crop) ────────────────────
        self.hist: np.ndarray | None = None

    # ── Kalman steps ─────────────────────────────────────────────────────────

    def predict(self) -> None:
        self._prev_lost = self.lost
        if self.lost > 0:
            # Freeze velocity so the box stays at last known position
            # instead of drifting while the person is not detected.
            self._x[4] = 0.0
            self._x[5] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, bbox: tuple[int, int, int, int]) -> None:
        z = np.array(_bbox_to_cwh(bbox), dtype=float)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ self._H) @ self._P
        self.lost  = 0
        self.hits += 1

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self._x[:4]
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    @property
    def just_returned(self) -> bool:
        """True the first frame a previously-lost track is matched again."""
        return self._prev_lost > 0 and self.lost == 0

    @property
    def just_lost(self) -> bool:
        return self._prev_lost == 0 and self.lost > 0


# ══════════════════════════════════════════════════════════ Detector wrapper ═

class _Detector:
    """
    Pluggable person detector. Set DETECTOR_MODE at the top of this file to switch.

    All modes return list[(x1,y1,x2,y2)] bounding boxes for detected persons.
    The tracker above is detector-agnostic — swapping modes requires no other changes.

    Latency measured at 480×270, CPU only (no GPU):
      yolo       ~17ms/call  →  ~6ms avg with DETECT_EVERY=3
      mediapipe  ~15ms/call  →  ~5ms avg with DETECT_EVERY=3
      mog2       ~2ms/call   →  ~2ms avg (runs every frame)

    Note: OpenCV HOG people detector was removed in OpenCV 5 and is not supported.
    """

    def __init__(self, mode: str = "yolo") -> None:
        self._mode = mode
        self._init(mode)

    def _init(self, mode: str) -> None:
        if mode == "yolo":
            self._init_yolo()
        elif mode == "mediapipe":
            self._init_mediapipe()
        else:
            self._mode = "mog2"
            self._init_mog2()

    # ── YOLO ─────────────────────────────────────────────────────────────────

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")
            self._yolo(np.zeros((64, 64, 3), dtype=np.uint8),
                       classes=[0], verbose=False)        # warm-up
            print("[identity] detector: YOLOv8-nano (imgsz=320)")
        except Exception as e:
            print(f"[identity] YOLO failed ({e}), falling back to MOG2")
            self._mode = "mog2"
            self._init_mog2()

    def _detect_yolo(self, frame: np.ndarray) -> list[tuple]:
        results = self._yolo(frame, classes=[0], verbose=False,
                             conf=0.30, imgsz=320)
        h, w = frame.shape[:2]
        min_area = h * w * 0.03   # must cover ≥3% of frame area
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bh, bw = y2 - y1, x2 - x1
                if bw < 1 or bh < 1:
                    continue
                if bh / bw < 0.6:   # reject horizontal crops (raised arms, etc.)
                    continue
                if bh * bw < min_area:  # reject tiny detections
                    continue
                boxes.append((x1, y1, x2, y2))
        return boxes

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    # Uses the Tasks API (mediapipe ≥ 0.10) with num_poses=2 so both players
    # are detected in a single inference pass.

    _MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                      "pose_landmarker/pose_landmarker_lite/float16/latest/"
                      "pose_landmarker_lite.task")
    _MP_MODEL_FILE = "pose_landmarker_lite.task"

    def _init_mediapipe(self) -> None:
        import os, urllib.request
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision as mp_vis

            if not os.path.exists(self._MP_MODEL_FILE):
                print(f"[identity] downloading MediaPipe model (~3 MB)…")
                urllib.request.urlretrieve(self._MP_MODEL_URL, self._MP_MODEL_FILE)

            opts = mp_vis.PoseLandmarkerOptions(
                base_options=mp_py.BaseOptions(
                    model_asset_path=self._MP_MODEL_FILE
                ),
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._mp_landmarker = mp_vis.PoseLandmarker.create_from_options(opts)
            print("[identity] detector: MediaPipe PoseLandmarker (2-pose lite)")
        except Exception as e:
            print(f"[identity] MediaPipe Tasks API failed ({e}), falling back to MOG2")
            self._mode = "mog2"
            self._init_mog2()

    def _detect_mediapipe(self, frame: np.ndarray) -> list[tuple]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = self._mp_landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return []
        h, w = frame.shape[:2]
        boxes = []
        for pose_lms in result.pose_landmarks:
            xs = [lm.x * w for lm in pose_lms]
            ys = [lm.y * h for lm in pose_lms]
            pad = 20
            x1 = max(0,  int(min(xs)) - pad)
            y1 = max(0,  int(min(ys)) - pad)
            x2 = min(w,  int(max(xs)) + pad)
            y2 = min(h,  int(max(ys)) + pad)
            boxes.append((x1, y1, x2, y2))
        return boxes

    # ── MOG2 (background subtraction) ────────────────────────────────────────
    # Fastest option (~2ms). Best for a fixed-camera setup like this game.
    # Learns the static background over the first ~2 s, then detects movement.

    _MOG2_WARMUP = 50   # frames to learn the background before detecting

    def _init_mog2(self) -> None:
        # history=500: takes ~500 frames to absorb a static object (not players)
        # varThreshold=16: sensitive enough to catch near-static people
        self._mog2       = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self._mog2_frame = 0
        self._kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        print("[identity] detector: MOG2 background subtraction "
              f"(step AWAY from camera for first {self._MOG2_WARMUP} frames)")

    def _detect_mog2(self, frame: np.ndarray) -> list[tuple]:
        self._mog2_frame += 1

        # During warmup let MOG2 learn freely.
        # After warmup use a very low learning rate so standing players
        # are not gradually absorbed into the background model.
        lr = -1 if self._mog2_frame < self._MOG2_WARMUP else 0.0005
        mask = self._mog2.apply(frame, learningRate=lr)

        if self._mog2_frame < self._MOG2_WARMUP:
            return []   # still learning background

        # Clean up noise: open removes small blobs, dilate merges nearby blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=3)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Minimum area ~ a person at 480×270
        MIN_AREA = frame.shape[0] * frame.shape[1] * 0.01   # 1% of frame
        raw = []
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                raw.append((x, y, x + w, y + h))

        return self._merge_nearby(raw)

    def _merge_nearby(self, boxes: list) -> list:
        """Merge overlapping / adjacent foreground blobs into one person box."""
        if len(boxes) <= 1:
            return boxes
        merged, used = [], [False] * len(boxes)
        for i, bi in enumerate(boxes):
            if used[i]:
                continue
            x1, y1, x2, y2 = bi
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bj = boxes[j]
                if _iou(bi, bj) > 0.05:          # any overlap → same person
                    x1 = min(x1, bj[0]); y1 = min(y1, bj[1])
                    x2 = max(x2, bj[2]); y2 = max(y2, bj[3])
                    used[j] = True
            merged.append((x1, y1, x2, y2))
            used[i] = True
        return merged

    # ── dispatch ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._mode == "yolo":
            return self._detect_yolo(frame)
        if self._mode == "mediapipe":
            return self._detect_mediapipe(frame)
        return self._detect_mog2(frame)


# ══════════════════════════════════════════════════════════ Helpers ══════════

def _bbox_to_cwh(b: tuple) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return (x1+x2)/2., (y1+y2)/2., float(x2-x1), float(y2-y1)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aA    = max(1, (ax2-ax1)*(ay2-ay1))
    bA    = max(1, (bx2-bx1)*(by2-by1))
    return inter / (aA + bA - inter)


def _hsv_hist(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """16×8 bin HSV histogram — the player fingerprint."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, bbox[0]), max(0, bbox[1]),
        min(w, bbox[2]), min(h, bbox[3]),
    )
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(128, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _hist_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _clamp(bbox: tuple, h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


def _hungarian_2x2(cost: np.ndarray) -> list[tuple[int, int]]:
    """
    Optimal assignment for up to a 2×2 cost matrix, no scipy needed.
    Falls back to scipy for larger matrices (shouldn't happen with 2 players).

    Returns list of (row, col) pairs — every returned pair is a candidate
    assignment; the caller decides which ones to accept based on cost threshold.
    """
    r, c = cost.shape
    if r == 0 or c == 0:
        return []

    if r == 1 and c == 1:
        return [(0, 0)]

    if r == 1 and c == 2:
        # 1 track, 2 detections — assign to the lower-cost detection
        return [(0, 0)] if cost[0, 0] <= cost[0, 1] else [(0, 1)]

    if r == 2 and c == 1:
        # 2 tracks, 1 detection — assign to the lower-cost track
        return [(0, 0)] if cost[0, 0] <= cost[1, 0] else [(1, 0)]

    if r == 2 and c == 2:
        # Try both full assignments, pick minimum total cost
        if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0]:
            return [(0, 0), (1, 1)]
        return [(0, 1), (1, 0)]

    # Larger — use scipy
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


# ══════════════════════════════════════════════════════════ Main module ══════

class Module(Base):
    name = "identity"

    def __init__(self) -> None:
        import os
        mode = os.environ.get("IDENTITY_DETECTOR", DETECTOR_MODE)
        self.detector  = _Detector(mode=mode)
        self.tracks: dict[int, _KalmanTrack] = {}
        self._frame_idx = 0
        # MOG2 runs every frame (it IS the tracking, not just detection)
        self._always_detect = (self.detector._mode == "mog2")

    def reset(self) -> None:
        self.tracks     = {}
        self._frame_idx = 0

    # ── entry point ──────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, state: GameState) -> None:
        h, w = frame.shape[:2]

        # 1. Kalman predict — moves every track forward one frame
        for t in self.tracks.values():
            t.predict()

        # 2. Run detector when due, or whenever all tracks are absent
        run_detect = (
            self._always_detect                         # MOG2 must run every frame
            or self._frame_idx % DETECT_EVERY == 0
            or len(self.tracks) == 0                    # no one initialised yet
            or any(t.lost > 5 for t in self.tracks.values())
        )
        detections: list[tuple] = []
        if run_detect:
            raw = self.detector.detect(frame)
            # Keep only the two largest bounding boxes
            raw.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
            detections = [_clamp(b, h, w) for b in raw[:2]]

        # 3. First initialisation — assign leftmost → P0, rightmost → P1
        if not self.tracks:
            if detections:
                detections.sort(key=lambda b: (b[0]+b[2]) / 2)
                for pid, det in enumerate(detections[:2]):
                    t = _KalmanTrack(det, pid)
                    t.hist = _hsv_hist(frame, det)
                    self.tracks[pid] = t
            else:
                state.phase   = "waiting"
                state.message = "Waiting for players…"
            self._frame_idx += 1
            self._write_state(frame, state, h, w)
            return

        # 4. Associate detections to tracks (two-stage)
        if detections:
            matched, unmatched_trk, unmatched_det = self._associate(
                detections, frame
            )

            for tid, did in matched.items():
                bbox = detections[did]
                self.tracks[tid].update(bbox)
                new_hist = _hsv_hist(frame, bbox)
                old_hist = self.tracks[tid].hist
                if old_hist is not None:
                    self.tracks[tid].hist = (
                        (1 - HIST_ALPHA) * old_hist + HIST_ALPHA * new_hist
                    )
                else:
                    self.tracks[tid].hist = new_hist

            for tid in unmatched_trk:
                self.tracks[tid].lost += 1

            for did in unmatched_det:
                self._try_reenter(detections[did], frame)

        elif run_detect:
            # Detector ran but found nobody — all tracks are truly missing
            for t in self.tracks.values():
                t.lost += 1
        else:
            # Detector was skipped this frame (DETECT_EVERY cadence).
            # Don't penalise a present player just because we didn't check.
            # Tracks already known to be lost keep counting every frame.
            for t in self.tracks.values():
                if t.lost > 0:
                    t.lost += 1

        # 5. Fire tracking events, then write players into state
        self._fire_events(state)
        self._write_state(frame, state, h, w)

        # Cap lost counter so it does not grow unboundedly
        for t in self.tracks.values():
            t.lost = min(t.lost, MAX_LOST)

        self._frame_idx += 1

    # ── two-stage association ─────────────────────────────────────────────────

    def _associate(
        self,
        detections: list[tuple],
        frame: np.ndarray,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """
        Stage 1 — IoU Hungarian.
            Low cost (high IoU) = same person, probably just moved a bit.

        Stage 2 — HSV histogram similarity.
            Used for any track that Stage 1 could not match confidently —
            crossing players, overlapping boxes during an attack.

        Returns:
            matched        {track_id: detection_index}
            unmatched_trk  track ids with no match
            unmatched_det  detection indices with no match
        """
        track_ids = list(self.tracks.keys())
        n_t, n_d  = len(track_ids), len(detections)

        if n_t == 0 or n_d == 0:
            return {}, track_ids[:], list(range(n_d))

        # ── Stage 1: IoU ─────────────────────────────────────────────────────
        iou_cost = np.zeros((n_t, n_d))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_cost[i, j] = 1.0 - _iou(self.tracks[tid].bbox, det)

        assignments = _hungarian_2x2(iou_cost)

        matched:       dict[int, int] = {}
        used_det:      set[int]       = set()
        unmatched_trk: list[int]      = []

        for r, c in assignments:
            tid = track_ids[r]
            if iou_cost[r, c] <= (1.0 - IOU_THRESH):   # good IoU match
                matched[tid] = c
                used_det.add(c)
            else:
                unmatched_trk.append(tid)

        # tracks that appeared in no assignment
        assigned_rows = {r for r, _ in assignments}
        for i, tid in enumerate(track_ids):
            if i not in assigned_rows and tid not in unmatched_trk:
                unmatched_trk.append(tid)

        unmatched_det = [j for j in range(n_d) if j not in used_det]

        # ── Stage 2: histogram ────────────────────────────────────────────────
        still_unmatched: list[int] = []
        for tid in unmatched_trk:
            if not unmatched_det:
                still_unmatched.append(tid)
                continue
            best_sim, best_did = -1.0, -1
            for did in unmatched_det:
                cand = _hsv_hist(frame, detections[did])
                sim  = _hist_sim(self.tracks[tid].hist, cand)
                if sim > best_sim:
                    best_sim, best_did = sim, did
            if best_sim >= HIST_THRESH:
                matched[tid] = best_did
                unmatched_det.remove(best_did)
            else:
                still_unmatched.append(tid)

        return matched, still_unmatched, unmatched_det

    # ── re-entry ──────────────────────────────────────────────────────────────

    def _try_reenter(self, bbox: tuple, frame: np.ndarray) -> None:
        """
        A detection was not matched to any active track.
        Compare its histogram against tracks that have been lost for a while —
        if it looks like the same player, reassign the identity.
        """
        cand_hist = _hsv_hist(frame, bbox)
        best_sim, best_tid = -1.0, -1
        for tid, t in self.tracks.items():
            if t.lost < 5:
                continue   # only try to revive actually-lost tracks
            sim = _hist_sim(t.hist, cand_hist)
            if sim > best_sim:
                best_sim, best_tid = sim, tid

        if best_tid >= 0 and best_sim >= HIST_THRESH:
            self.tracks[best_tid].update(bbox)
            new_hist = _hsv_hist(frame, bbox)
            self.tracks[best_tid].hist = (
                (1 - HIST_ALPHA) * self.tracks[best_tid].hist
                + HIST_ALPHA * new_hist
            )

    # ── events ────────────────────────────────────────────────────────────────

    def _fire_events(self, state: GameState) -> None:
        for t in self.tracks.values():
            if t.just_lost:
                state.events.append(
                    GameEvent(kind="player_lost", player_id=t.player_id)
                )
            if t.just_returned:
                state.events.append(
                    GameEvent(kind="player_returned", player_id=t.player_id)
                )

    # ── write state ───────────────────────────────────────────────────────────

    def _write_state(
        self, frame: np.ndarray, state: GameState, h: int, w: int
    ) -> None:
        """
        Populate state.players.
        Each Player gets a stable player_id, the clamped bounding box,
        frames_missing, and a colour so other modules don't have to agree on
        colours independently.
        """
        for tid, t in self.tracks.items():
            if t.lost >= MAX_LOST:
                continue  # fully evicted; kept in self.tracks for re-entry only

            bbox   = _clamp(t.bbox, h, w)
            colour = COLOURS.get(tid, (200, 200, 200))
            x1, y1, x2, y2 = bbox

            state.players.append(Player(
                player_id     = tid,
                bbox          = bbox if t.lost == 0 else None,
                frames_missing= t.lost,
                colour        = colour,
            ))

            # After LOST_HIDE frames the box disappears entirely.
            # Between 0 and LOST_HIDE it fades steadily.
            if t.lost >= LOST_HIDE:
                continue

            alpha = max(0.25, 1.0 - t.lost / LOST_HIDE)
            draw_colour = tuple(int(c * alpha) for c in colour)

            state.draw.append(DrawCommand(
                kind="rect",
                x=float(x1), y=float(y1), x2=float(x2), y2=float(y2),
                colour=draw_colour, thickness=2, layer=1,
            ))
            state.draw.append(DrawCommand(
                kind="text",
                x=float(x1), y=float(max(y1 - 8, 12)),
                text=f"P{tid}", colour=draw_colour, thickness=1, layer=2,
            ))
