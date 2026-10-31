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

import time
import numpy as np
import cv2

from game_types import BBox, DrawCommand, GameEvent, GameState, Module as Base, Player, player_label

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
        from pathlib import Path
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision as mp_vis

            # Prefer shared project models/ (same file pose uses), then cwd.
            root_models = Path(__file__).resolve().parents[2] / "models"
            candidates = (
                root_models / "pose_landmarker_lite.task",
                root_models / "pose_landmarker_full.task",
                Path(self._MP_MODEL_FILE),
            )
            model_path = next((p for p in candidates if p.exists()), None)
            if model_path is None:
                model_path = root_models / "pose_landmarker_lite.task"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                print("[identity] downloading MediaPipe model (~3 MB)…")
                urllib.request.urlretrieve(self._MP_MODEL_URL, str(model_path))

            opts = mp_vis.PoseLandmarkerOptions(
                base_options=mp_py.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp_vis.RunningMode.IMAGE,
                num_poses=2,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
            )
            self._mp_landmarker = mp_vis.PoseLandmarker.create_from_options(opts)
            print(f"[identity] detector: MediaPipe PoseLandmarker "
                  f"(2-pose, {model_path.name})")
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
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self._m_frames        = 0   # total process() calls
        self._m_detect_calls  = 0   # times detector actually ran
        self._m_player_frames: dict[int, int] = {}   # frames each pid was present (lost==0)
        self._m_lost_frames:   dict[int, int] = {}   # frames each pid was lost (lost>0)
        # SORT: re-entry (successful histogram match on re-appearance)
        self._m_reentry       = 0
        # SORT: ID switches — re-entry attempted but histogram match failed → new ID
        self._m_id_switches   = 0
        # SORT: track fragments — each time a present track goes lost (gap opens)
        self._m_fragments: dict[int, int] = {}
        # MOTP proxy: centroid jitter — Welford online variance of cx,cy per pid
        # stored as {pid: [n, mean_x, M2_x, mean_y, M2_y]}
        self._m_jitter: dict[int, list] = {}
        # Re-ID time: frame when each pid went lost → compute gap on return
        self._m_lost_at: dict[int, int] = {}
        # Re-ID time samples (frames from lost to returned), per pid
        self._m_reid_times: dict[int, list] = {}
        # HopTrack: track continuity (AssA proxy)
        # consecutive present-frames in the current run, per pid
        self._m_current_run: dict[int, int] = {}
        # completed run lengths per pid (recorded when a gap opens)
        self._m_run_lengths: dict[int, list] = {}
        # HopTrack: simultaneous detection — frames where both P0 and P1 present
        self._m_both_present = 0
        # timing: wall-clock start and cumulative processing time
        self._m_wall_start: float = time.time()
        self._m_proc_total_ms: float = 0.0

    def _print_metrics(self) -> None:
        import os
        if os.environ.get("IDENTITY_METRICS") != "1":
            return
        mode = self.detector._mode
        total = self._m_frames
        if total == 0:
            return

        elapsed_s   = max(time.time() - self._m_wall_start, 0.001)
        elapsed_min = elapsed_s / 60.0
        mean_lat_ms = self._m_proc_total_ms / total
        mean_fps    = 1000.0 / max(mean_lat_ms, 0.001)

        lines: list[str] = []
        add = lines.append

        add("")
        add(f"── identity metrics ({mode}) ──────────────────────────────")
        add(f"  timestamp         : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        add(f"  detector backend  : {mode}")
        add(f"  total frames      : {total}")
        add(f"  elapsed time      : {elapsed_s:.1f} s  ({elapsed_min:.2f} min)")
        add(f"  detector calls    : {self._m_detect_calls}  "
            f"(1 every {total/max(1,self._m_detect_calls):.1f} frames)")
        add(f"  mean latency      : {mean_lat_ms:.2f} ms/frame  "
            f"({mean_fps:.1f} FPS equivalent)  [processing time]")
        add("")

        all_pids = sorted(
            set(self._m_player_frames) | set(self._m_lost_frames)
        )

        add("  [SORT metrics]")
        for pid in all_pids:
            pf = self._m_player_frames.get(pid, 0)
            lf = self._m_lost_frames.get(pid, 0)
            tracked = pf + lf
            if tracked == 0:
                continue
            rate = 100.0 * pf / tracked
            frags = self._m_fragments.get(pid, 0)
            add(f"  {player_label(pid)} detection rate  : {rate:.1f}%  "
                f"({pf} present / {lf} lost / {tracked} total)")
            add(f"  {player_label(pid)} fragments       : {frags}  "
                f"(track gaps opened)")
            j = self._m_jitter.get(pid)
            if j and j[0] >= 2:
                std_x = (j[2] / j[0]) ** 0.5
                std_y = (j[4] / j[0]) ** 0.5
                jitter_px = (std_x**2 + std_y**2) ** 0.5
                add(f"  {player_label(pid)} centroid jitter : {jitter_px:.1f} px RMS  "
                    f"(MOTP proxy, still-player)")
        for pid in all_pids:
            times = self._m_reid_times.get(pid, [])
            if times:
                mean_t = sum(times) / len(times)
                worst  = max(times)
                add(f"  {player_label(pid)} Re-ID time     : mean {mean_t:.1f} frames, "
                    f"worst {worst} frames  ({len(times)} exit-return trials)")

        add("")
        add(f"  ID switches       : {self._m_id_switches}  "
            f"({self._m_id_switches / elapsed_min:.2f} /min)  "
            f"[re-entry failed → new ID]")
        add(f"  re-entry matches  : {self._m_reentry}  "
            f"[same ID recovered on re-appearance]")
        if self._m_reentry + self._m_id_switches > 0:
            idf1_proxy = 100.0 * self._m_reentry / (
                self._m_reentry + self._m_id_switches
            )
            add(f"  IDF1 proxy        : {idf1_proxy:.1f}%  "
                f"[re-entry accuracy; 100% = no identity loss]")

        # ── HopTrack metrics ────────────────────────────────────────────────
        add("")
        add("  [HopTrack metrics]")
        for pid in all_pids:
            runs = list(self._m_run_lengths.get(pid, []))
            open_run = self._m_current_run.get(pid, 0)
            if open_run > 0:
                runs.append(open_run)
            if runs:
                mean_run = sum(runs) / len(runs)
                worst_run = max(runs)
                add(f"  {player_label(pid)} track continuity: mean {mean_run:.1f} frames, "
                    f"longest {worst_run} frames  (AssA proxy)")
        simul_rate = 100.0 * self._m_both_present / total
        add(f"  simultaneous det. : {simul_rate:.1f}%  "
            f"({self._m_both_present}/{total} frames both players visible)")
        hop_attempts = self._m_reentry + self._m_id_switches
        hop_rate = 100.0 * hop_attempts / max(total, 1)
        add(f"  hop attempt rate  : {hop_rate:.1f}%  "
            f"({hop_attempts} re-associations over {total} frames)")

        add("────────────────────────────────────────────────────────────")

        for line in lines:
            print(line)

    def reset(self) -> None:
        self._print_metrics()
        self.tracks     = {}
        self._frame_idx = 0
        self._reset_metrics()

    def __del__(self) -> None:
        self._print_metrics()

    # ── entry point ──────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, state: GameState) -> None:
        _proc_t0 = time.perf_counter()
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
            self._m_detect_calls += 1
        self._m_frames += 1

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
                if not self._try_reenter(detections[did], frame):
                    # Not a re-entry of a lost player — could be a genuinely new
                    # second player arriving after the game started.
                    active_ids = {
                        tid for tid, t in self.tracks.items()
                        if t.lost < MAX_LOST
                    }
                    for new_pid in range(2):
                        if new_pid not in active_ids:
                            t = _KalmanTrack(detections[did], new_pid)
                            t.hist = _hsv_hist(frame, detections[did])
                            self.tracks[new_pid] = t
                            break

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
        self._m_proc_total_ms += (time.perf_counter() - _proc_t0) * 1000.0

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

    def _try_reenter(self, bbox: tuple, frame: np.ndarray) -> bool:
        """
        A detection was not matched to any active track.
        Compare its histogram against tracks that have been lost for a while.
        Returns True if the detection was consumed (matched to a lost track),
        False if no lost track matched (caller should treat it as a new player).
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
            self._m_reentry += 1
            return True
        else:
            # Histogram match failed against all lost tracks.
            # If there were lost tracks to compare against, this is an ID switch
            # (SORT metric) — the player's appearance changed enough to lose them.
            if any(t.lost >= 5 for t in self.tracks.values()):
                self._m_id_switches += 1
            return False

    # ── events ────────────────────────────────────────────────────────────────

    def _fire_events(self, state: GameState) -> None:
        for t in self.tracks.values():
            if t.just_lost:
                state.events.append(
                    GameEvent(kind="player_lost", player_id=t.player_id)
                )
                # SORT fragmentation: each gap opening = one fragment
                pid = t.player_id
                self._m_fragments[pid] = self._m_fragments.get(pid, 0) + 1
                self._m_lost_at[pid] = self._m_frames
                # HopTrack AssA proxy: close the current continuous run
                run = self._m_current_run.pop(pid, 0)
                if run > 0:
                    self._m_run_lengths.setdefault(pid, []).append(run)
            if t.just_returned:
                state.events.append(
                    GameEvent(kind="player_returned", player_id=t.player_id)
                )
                pid = t.player_id
                if pid in self._m_lost_at:
                    gap = self._m_frames - self._m_lost_at.pop(pid)
                    self._m_reid_times.setdefault(pid, []).append(gap)

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

            # BBox is the shared contract with Task 2 (pose): x0,y0,x1,y1
            # in full-frame pixels. Clear the box while lost so pose skips.
            player_bbox = (
                BBox(x0=float(x1), y0=float(y1), x1=float(x2), y1=float(y2))
                if t.lost == 0 else None
            )
            state.players.append(Player(
                player_id     = tid,
                bbox          = player_bbox,
                frames_missing= t.lost,
                colour        = colour,
            ))
            if t.lost == 0:
                self._m_player_frames[tid] = self._m_player_frames.get(tid, 0) + 1
                # HopTrack AssA proxy: extend current continuous run
                self._m_current_run[tid] = self._m_current_run.get(tid, 0) + 1
                # Welford online variance for centroid jitter (MOTP proxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if tid not in self._m_jitter:
                    self._m_jitter[tid] = [0, 0.0, 0.0, 0.0, 0.0]  # n,mx,M2x,my,M2y
                j = self._m_jitter[tid]
                j[0] += 1
                dx = cx - j[1]; j[1] += dx / j[0]; j[2] += dx * (cx - j[1])
                dy = cy - j[3]; j[3] += dy / j[0]; j[4] += dy * (cy - j[3])
            else:
                self._m_lost_frames[tid] = self._m_lost_frames.get(tid, 0) + 1

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
                text=player_label(tid), colour=draw_colour, thickness=1, layer=2,
            ))

        # HopTrack: simultaneous detection — both P0 and P1 present this frame
        present_pids = {
            t.player_id for t in self.tracks.values()
            if t.lost == 0 and t.player_id in (0, 1)
        }
        if 0 in present_pids and 1 in present_pids:
            self._m_both_present += 1
