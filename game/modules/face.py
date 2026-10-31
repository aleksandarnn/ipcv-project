"""
Task 1 - Face tracking and Augmented Effects
Owner: Shilpa John

Pipeline (runs second: identity -> face -> pose -> control -> render):
  raw frame
    → BlazeFace face detection
    → Kalman prediction of each face centre
    → match faces to Task 3 players (body box + distance gate)
    → Kalman correction, EMA smoothing of size and tilt
    → short misses bridged for up to 5 frames
    → headband + HP bar as draw commands

Contract
--------
**Input** (via ``state.players``):
  - ``player_id``, ``player.bbox``, ``frames_missing`` from Task 3
  - ``player.health`` from Task 4, read from the previous frame's
    players (control runs after face, so one frame late)

**Output**:
  - ``player.face`` → ``Face(cx, cy, w, h, angle, confidence)``, only when
    the face was really detected this frame; otherwise left as ``None``
  - ``state.draw`` → headband lines/circles (layer 1), HP bar and
    "P0  85" label (layer 2); the renderer (Task 5) draws them

**Model**: ``models/blaze_face_full_range.tflite`` in the project root.

**Debug**: run ``python main.py --debug`` to also draw the face box
(green = detected this frame, cyan = predicted).

**Metrics**: run ``python main.py --metrics`` to print the match rate and
the raw vs filtered jitter when the game exits. Off by default.

Design references: Kalman filter (Welch & Bishop, "An Introduction to the
Kalman Filter", UNC TR 95-041).
"""
from pathlib import Path
import atexit
import time
import math

import cv2
import mediapipe as mp
import numpy as np

from game_types import Module as Base, DrawCommand, Face

# BlazeFace is used for face detection. Its bounding box and keypoints
# provide the position, size and approximate rotation needed for the headband.
# A detailed face mesh is not required for this 2D effect.
MODEL_FILE = "blaze_face_full_range.tflite"

# Kalman filter parameters
# Higher velocity noise allows the filter to respond to sudden movement.
Q_POS = 1.0
Q_VEL = 2000.0
R_MEAS = 9.0

# Headband cloth colours (BGR)
BAND_COLOURS = ((125, 60, 30), (40, 30, 165), (50, 105, 45), (40, 130, 175))

# Health bar above the head. Health comes from player.health (task 4).
HP_MAX = 100

# Allow some margin when matching faces to body boxes
FACE_ABOVE_BOX = 0.75
SIDE_MARGIN = 0.05

# Maximum matching distance, measured in face widths
GATE_FACE_WIDTHS = 1.5

class Module(Base):
    name = "face"

    def __init__(self, model_file: str = MODEL_FILE):
        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / model_file 
        )

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=0.5
        )

        self.detector = (
            mp.tasks.vision.FaceDetector.create_from_options(options)
        )

        # Timing for MediaPipe and frame-to-frame tracking
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1
        self.last_frame_time = None

        # Controls EMA smoothing of face dimensions, rotation and approximate yaw.
        self.alpha = 0.5
        # controls how long the tracker keeps predicting when a detection is missing.
        self.max_missing_frames = 5

        # main.py sets this from --debug
        self.debug = False

        # One Kalman filter and face history per player
        self.filters = {}
        self.faces = {}            # last tracked face per player (measured or predicted)
        self.measured = {}         # last MEASURED face centre per player
        self.missing = {}

        # Headband effect. Set show_effects = False to see the plain
        # tracking boxes instead (with --debug you get both).
        self.show_effects = True
        self.yaw = {}              # player id -> smoothed head turn, -1 .. 1
        self.hp_shown = {}         # player id -> health shown on the bar (eases)
        # Players of the previous frame. Identity builds NEW Player objects
        # every frame (health = 100) and control (task 4) sets health AFTER
        # face has run, so the real health is read from last frame's players.
        self._prev_players = []
        self.last_health = {}      # player id -> health from the previous frame

        # counters for the report
        self.stats = {"frames": 0, "detected": 0, "matched": 0, "reinit": 0}

        # Metrics (match rate, jitter): off by default. main.py sets
        # self.metrics = True for "python main.py --metrics".
        self.metrics = False

        # Jitter: per player, frame-to-frame movement of the raw detected
        # centre and of the filtered centre, over consecutive matched frames.
        # Only collected when metrics are on.
        self.jitter_raw = {}
        self.jitter_filtered = {}
        self._last_centres = {}
        # Alignment error: per player, distance between the detected face
        # centre and the filtered centre (where the effect is drawn)
        self.align_px = {}
        self.align_fw = {}
        # Dynamic jitter: change of movement between frames (second
        # difference); smooth motion gives ~0, so it also works while moving
        self.shake_raw = {}
        self.shake_filtered = {}
        self.two_face = {"frames": 0, "detected": 0, "matched": 0}
        # registered always, prints only if metrics are on (main.py sets the
        # flag after the module is created)
        atexit.register(self.print_metrics)

    def create_filter(self, face):
        """Track face centre and velocity: x, y, vx, vy."""

        kf = cv2.KalmanFilter(4, 2)

        kf.transitionMatrix = np.eye(4, dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        kf.processNoiseCov = np.diag(
            [Q_POS, Q_POS, Q_VEL, Q_VEL]
        ).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R_MEAS
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10

        kf.statePost = np.array([
            [face.cx],
            [face.cy],
            [0],
            [0]
        ], dtype=np.float32)

        return kf

    def get_face(self, detection, width, height):
        """Convert a MediaPipe detection into (Face, at_edge, yaw).

        The box is NOT clipped to the frame: clipping a face that is half out
        of the picture shrinks the box and pulls its centre inwards. Instead
        at_edge tells smooth_geometry to keep the previous size.
        """

        box = detection.bounding_box

        x0 = box.origin_x
        y0 = box.origin_y
        x1 = box.origin_x + box.width
        y1 = box.origin_y + box.height

        at_edge = x0 < 0 or y0 < 0 or x1 > width or y1 > height

        angle = 0.0
        yaw = 0.0 

        if len(detection.keypoints) >= 2:
            e1 = (detection.keypoints[0].x * width,
                  detection.keypoints[0].y * height)
            e2 = (detection.keypoints[1].x * width,
                  detection.keypoints[1].y * height)

            # Calculate head tilt from the eye positions
            left_eye, right_eye = sorted((e1, e2))

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]

            angle = math.degrees(math.atan2(dy, dx))

            # Yaw (left/right head turn): where the nose sits between the
            # eyes. Halfway = facing the camera, near one eye = turned.
            eye_dist = math.hypot(dx, dy)
            if len(detection.keypoints) >= 3 and eye_dist > 1:
                nose_x = detection.keypoints[2].x * width
                yaw = (nose_x - (left_eye[0] + right_eye[0]) / 2) / eye_dist
                yaw = max(-1.0, min(1.0, 2.0 * yaw))

        face = Face(
            cx=(x0 + x1) / 2,
            cy=(y0 + y1) / 2,
            w=x1 - x0,
            h=y1 - y0,
            angle=angle,
            confidence=detection.categories[0].score
        )

        return face, at_edge, yaw

    def smooth_geometry(self, face, previous, at_edge):
        """Smooth face width, height and tilt with an EMA. Returns (w, h, angle).
        """

        a = self.alpha

        # Use the shortest angular difference to avoid wraparound jumps
        angle_diff = (
            (face.angle - previous.angle + 180) % 360
        ) - 180

        if at_edge:
            # Keep the previous size when the face is partly outside the frame
            w, h = previous.w, previous.h
        else:
            w = a * face.w + (1 - a) * previous.w
            h = a * face.h + (1 - a) * previous.h

        return w, h, previous.angle + a * angle_diff

    def match_faces(self, detections, players, predictions):
        """Match faces to visible players.

        Returns (face index, player id, reinit) tuples.

        A face is only considered for a player if it belongs to that
        player's body box from identity (task 3), see face_in_box. Then:
          - distance is measured to the Kalman prediction AND to the last
            measured position, whichever is closer, so a lagging prediction
            cannot reject the real face;
          - the gate is GATE_FACE_WIDTHS face widths, not fixed pixels;
          - if a face fails the gate but is the ONLY face in that player's
            box, it is still accepted (identity already says it belongs to
            this player) and the filter is restarted on it.
        """

        candidates = []

        for player in players:
            if not player.visible or player.bbox is None:
                continue

            box = player.bbox
            pid = player.player_id

            in_box = []

            for i, face in enumerate(detections):
                if self.face_in_box(face, box):
                    in_box.append(i)

            for i in in_box:
                face = detections[i]

                if pid in self.faces:
                    targets = [predictions.get(pid), self.measured.get(pid)]
                    targets = [t for t in targets if t is not None]

                    distance = min(
                        math.hypot(face.cx - tx, face.cy - ty)
                        for tx, ty in targets
                    )

                    gate = GATE_FACE_WIDTHS * max(
                        face.w, self.faces[pid].w
                    )

                    if distance <= gate:
                        candidates.append((distance, i, pid, False))
                    elif len(in_box) == 1:
                        # lower priority than any face inside the gate
                        candidates.append((gate + distance, i, pid, True))
                else:
                    # new track: expect the head near the top of the body box
                    target = (
                        (box.x0 + box.x1) / 2,
                        box.y0 + box.h * 0.2
                    )

                    distance = math.hypot(
                        face.cx - target[0],
                        face.cy - target[1]
                    )

                    candidates.append((distance, i, pid, True))

        # Assign the closest valid pairs without duplicates
        candidates.sort()

        assignments = []
        used_faces = set()
        used_players = set()

        for _, i, pid, reinit in candidates:
            if i in used_faces or pid in used_players:
                continue

            assignments.append((i, pid, reinit))
            used_faces.add(i)
            used_players.add(pid)

        return assignments

    @staticmethod
    def face_in_box(face, box):
        """Check whether the face centre is within the player's body region,
        allowing a small margin at the sides and above the box.
        """
        side = SIDE_MARGIN * box.w
        return (
            box.x0 - side <= face.cx <= box.x1 + side
            and box.y0 - FACE_ABOVE_BOX * face.h <= face.cy <= box.y1
        )

    def update_face(self, pid, face, at_edge, reinit):
        """Update a player's face track using Kalman filtering and smoothing."""

        if reinit or pid not in self.filters: # Initialize or restart the Kalman filter
            if pid in self.filters:
                self.stats["reinit"] += 1
            self.filters[pid] = self.create_filter(face)
            previous = self.faces.get(pid)
            if previous is not None and at_edge:
                face = Face(face.cx, face.cy, previous.w, previous.h,
                            face.angle, face.confidence)
            tracked = face
        else: # Correct the position and smooth the face geometry
            measurement = np.array([
                [face.cx],
                [face.cy]
            ], dtype=np.float32)

            corrected = self.filters[pid].correct(measurement)

            w, h, angle = self.smooth_geometry(face, self.faces[pid], at_edge)

            tracked = Face(
                cx=float(corrected[0, 0]),
                cy=float(corrected[1, 0]),
                w=w,
                h=h,
                angle=angle,
                confidence=face.confidence
            )

        self.faces[pid] = tracked
        self.measured[pid] = (face.cx, face.cy)
        self.missing[pid] = 0
        if self.metrics:
            self.record_jitter(pid, (face.cx, face.cy),
                               (tracked.cx, tracked.cy), reinit)
            self.record_alignment(pid, (face.cx, face.cy),
                                  (tracked.cx, tracked.cy), face.w)

        return tracked

    def record_jitter(self, pid, raw, filtered, reinit):
        """Store how far the raw and the filtered face centre moved since the
        previous frame. A restarted track starts a new sequence."""
        history = [] if reinit else self._last_centres.get(pid, [])
        if history:
            (rx, ry), (fx, fy) = history[-1]
            self.jitter_raw.setdefault(pid, []).append(
                math.hypot(raw[0] - rx, raw[1] - ry))
            self.jitter_filtered.setdefault(pid, []).append(
                math.hypot(filtered[0] - fx, filtered[1] - fy))
        if len(history) == 2:
            # p(t) - 2 p(t-1) + p(t-2): zero for steady movement
            (r2x, r2y), (f2x, f2y) = history[0]
            self.shake_raw.setdefault(pid, []).append(
                math.hypot(raw[0] - 2 * rx + r2x, raw[1] - 2 * ry + r2y))
            self.shake_filtered.setdefault(pid, []).append(
                math.hypot(filtered[0] - 2 * fx + f2x,
                           filtered[1] - 2 * fy + f2y))
        self._last_centres[pid] = (history + [(raw, filtered)])[-2:]

    def record_alignment(self, pid, raw, filtered, width):
        """Store the distance between the detected face centre and the
        filtered centre where the effect is drawn, in px and face widths."""
        d = math.hypot(filtered[0] - raw[0], filtered[1] - raw[1])
        self.align_px.setdefault(pid, []).append(d)
        if width > 0:
            self.align_fw.setdefault(pid, []).append(d / width)

    def print_metrics(self):
        """Match rate and jitter, printed at exit when metrics are on.

        Match rate = detected faces attached to a player / detected faces.
        Jitter = mean frame-to-frame movement of the face centre; measure it
        with the players standing still, so the movement is detector noise.
        """
        s = self.stats
        if not self.metrics or s["frames"] == 0:
            return
        rate = 100.0 * s["matched"] / s["detected"] if s["detected"] else 0.0
        print()
        print("FACE MODULE REPORT (task 1)")
        print(f"frames          : {s['frames']}")
        print(f"faces detected  : {s['detected']}")
        print(f"faces matched   : {s['matched']}  ({rate:.1f} % match rate)")
        print(f"filter restarts : {s['reinit']}")
        t = self.two_face
        if t["frames"]:
            print(f"two-player frames: {t['frames']}, both faces detected "
                  f"{100.0 * t['detected'] / t['frames']:.1f} %, both matched "
                  f"{100.0 * t['matched'] / t['frames']:.1f} %")
        print("jitter, mean frame-to-frame centre movement (stand still to measure):")
        for pid in sorted(self.jitter_raw):
            raw = self.jitter_raw[pid]
            filt = self.jitter_filtered[pid]
            r = sum(raw) / len(raw)
            f = sum(filt) / len(filt)
            less = 100.0 * (1 - f / r) if r > 0 else 0.0
            print(f"   P{pid}: raw {r:.2f} px, filtered {f:.2f} px "
                  f"({less:.0f} % less, {len(raw)} frames)")
        print("dynamic jitter, change of movement per frame (valid while moving):")
        for pid in sorted(self.shake_raw):
            raw = self.shake_raw[pid]
            filt = self.shake_filtered[pid]
            r = sum(raw) / len(raw)
            f = sum(filt) / len(filt)
            less = 100.0 * (1 - f / r) if r > 0 else 0.0
            print(f"   P{pid}: raw {r:.2f} px, filtered {f:.2f} px "
                  f"({less:.0f} % less, {len(raw)} frames)")
        print("alignment error, detected face centre vs drawn effect centre:")
        for pid in sorted(self.align_px):
            px = self.align_px[pid]
            fw = self.align_fw.get(pid) or [0.0]
            print(f"   P{pid}: mean {sum(px) / len(px):.2f} px "
                  f"({sum(fw) / len(fw):.3f} face widths), "
                  f"max {max(px):.1f} px, {len(px)} frames")

    def remove_track(self, pid):
        """Remove the stored face tracking data for a player."""
        self._last_centres.pop(pid, None)
        self.filters.pop(pid, None)
        self.faces.pop(pid, None)
        self.measured.pop(pid, None)
        self.missing.pop(pid, None)

    def draw_face(self, face, state, colour):
        """Draw a rectangle around the tracked face."""

        state.draw.append(
            DrawCommand(
                kind="rect",
                x=int(face.cx - face.w / 2),
                y=int(face.cy - face.h / 2),
                x2=int(face.cx + face.w / 2),
                y2=int(face.cy + face.h / 2),
                colour=colour,
                thickness=2
            )
        )

    def process(self, frame, state):
        """Detect and track player faces, update their positions and add visual effects."""
        height, width = frame.shape[:2]

        now = time.monotonic()

        if self.last_frame_time is None:
            dt = 1 / 30
        else:
            dt = max(0.001, min(now - self.last_frame_time, 0.2))

        self.last_frame_time = now
        t = now - self.start_time

        # Store health values from the previous frame
        self.last_health = {
            p.player_id: p.health for p in self._prev_players
        }

        # Clear previous frame's face results
        for player in state.players:
            player.face = None

            if not player.visible:
                self.remove_track(player.player_id)

        # Remove tracks for players no longer reported by identity
        known = {p.player_id for p in state.players}
        for pid in list(self.faces):
            if pid not in known:
                self.remove_track(pid)

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Ensure timestamps increase for MediaPipe video mode
        timestamp_ms = max(
            int((now - self.start_time) * 1000),
            self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            image, timestamp_ms
        )

        found = [
            self.get_face(d, width, height)
            for d in result.detections
        ]
        detections = [f[0] for f in found]
        edges = [f[1] for f in found]
        yaws = [f[2] for f in found]

        # Predict existing face positions
        predictions = {}

        for pid, kf in self.filters.items():
            kf.transitionMatrix[0, 2] = dt
            kf.transitionMatrix[1, 3] = dt

            prediction = kf.predict()

            predictions[pid] = (
                float(prediction[0, 0]),
                float(prediction[1, 0])
            )

        # Match detections to players
        assignments = self.match_faces(
            detections, state.players, predictions
        )

        self.stats["frames"] += 1
        self.stats["detected"] += len(detections)
        self.stats["matched"] += len(assignments)
        if self.metrics:
            # Two-face detection rate: frames where identity sees two
            # players, and how often both faces were detected / matched
            if sum(1 for p in state.players if p.visible) >= 2:
                self.two_face["frames"] += 1
                if len(detections) >= 2:
                    self.two_face["detected"] += 1
                if len(assignments) >= 2:
                    self.two_face["matched"] += 1

        if self.debug and len(assignments) < len(detections):
            print(
                f"[face] detected {len(detections)}, "
                f"matched {len(assignments)}"
            )

        player_lookup = {
            p.player_id: p for p in state.players
        }

        # Update matched face tracks and smooth yaw estimates
        for i, pid, reinit in assignments:
            player_lookup[pid].face = self.update_face(
                pid, detections[i], edges[i], reinit
            )
            old = self.yaw.get(pid, yaws[i])
            self.yaw[pid] = self.alpha * yaws[i] + (1 - self.alpha) * old

        # Use predicted positions for temporarily missing faces
        for player in state.players:
            pid = player.player_id

            if not player.visible or player.face is not None:
                continue

            if pid not in self.faces:
                continue

            self.missing[pid] += 1
            # no measurement this frame: the jitter sequence starts again
            self._last_centres.pop(pid, None)

            if self.missing[pid] > self.max_missing_frames:
                self.remove_track(pid)
                continue

            previous = self.faces[pid]
            cx, cy = predictions[pid]

            self.faces[pid] = Face(
                cx=cx,
                cy=cy,
                w=previous.w,
                h=previous.h,
                angle=previous.angle,
                confidence=previous.confidence
            )

        # Draw effects and optional face tracking boxes
        for player in state.players:
            pid = player.player_id

            if not player.visible or pid not in self.faces:
                continue

            seen = self.missing.get(pid, 0) == 0

            if self.show_effects:
                self.draw_headband(pid, state, t)
                self.draw_hp_bar(player, state, width, height)

            if not self.show_effects or self.debug:
                # green = detected this frame, cyan = predicted
                colour = (0, 255, 0) if seen else (255, 255, 0)
                self.draw_face(self.faces[pid], state, colour)

        # Save player references for health updates in the next frame
        self._prev_players = state.players

    def draw_hp_bar(self, player, state, width, height):
        """Extra game effect (not on the face): a health bar floating above
        the player's head that follows them around.

        - Health is kept by task 4, which runs
          after face, so it is read from the previous frame (33 ms late)
        - colour goes green -> yellow -> red as health drops
        - border in the player's headband colour, so you see whose bar it is
        """
        pid = player.player_id
        face = self.faces[pid]
        w = face.w

        # Use previous-frame health when the current value is at its default
        health = player.health
        if health == HP_MAX and pid in self.last_health:
            health = self.last_health[pid]
        hp = max(0.0, min(float(HP_MAX), float(health)))
        shown = self.hp_shown.get(pid, hp)
        # the damage part shrinks towards the real value, faster when far off
        shown = hp if shown < hp else shown + (hp - shown) * 0.12
        self.hp_shown[pid] = shown

        # Position the bar above the face and keep it within the frame  
        bar_w = max(60.0, 1.1 * w)
        bar_h = max(6.0, 0.07 * w)
        x0 = face.cx - bar_w / 2
        y0 = face.cy - 1.05 * w - bar_h          # above the hair
        x0 = min(max(x0, 2.0), width - bar_w - 2.0)
        y0 = min(max(y0, 2.0), height - bar_h - 2.0)
        x1, y1 = x0 + bar_w, y0 + bar_h

        # Change the bar colour as health decreases
        frac = hp / HP_MAX
        if frac > 0.5:       # green -> yellow (BGR)
            fill = (40, 200, int(80 + 175 * (1 - frac) * 2))
        else:                # yellow -> red
            fill = (40, int(200 * frac * 2), 230)

        def rect(xa, ya, xb, yb, colour, thick=-1):
            state.draw.append(DrawCommand(
                kind="rect", x=xa, y=ya, x2=xb, y2=yb,
                colour=colour, thickness=thick, layer=2))

        rect(x0, y0, x1, y1, (25, 25, 25))    # empty part
        if shown > hp:
            rect(x0, y0, x0 + bar_w * shown / HP_MAX, y1, (200, 215, 235))  # recent damage
        if hp > 0:
            rect(x0, y0, x0 + bar_w * frac, y1, fill)
        border = BAND_COLOURS[pid % len(BAND_COLOURS)]
        rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1, border, 2)

        # Move the label beside the bar if there is no space above it
        scale = max(0.4, min(0.6, bar_w / 220))
        text_x, text_y = x0, y0 - 5
        if text_y < 14:
            text_y = y1
            text_x = x1 + 6 if x1 + 80 < width else x0 - 80
        state.draw.append(DrawCommand(
            kind="text", x=text_x, y=text_y,
            text=f"P{pid}  {int(round(hp))}",
            colour=(255, 255, 255), thickness=1, scale=scale, layer=2))

    def draw_headband(self, pid, state, t):
        """Draw a cloth samurai headband (hachimaki)  using the tracked face position,
           size and angle as part of face effect.
        """
        face = self.faces[pid]
        w = face.w
        yaw = self.yaw.get(pid, 0.0)
        a = math.radians(face.angle)
        cos_a, sin_a = math.cos(a), math.sin(a)

        # Convert face-relative coordinates to rotated image coordinate
        def pt(u, v, depth=0.0):
            u = u + depth * yaw
            return (face.cx + w * (u * cos_a - v * sin_a),
                    face.cy + w * (u * sin_a + v * cos_a))

        def line(p, q, colour, thick):
            state.draw.append(DrawCommand(
                kind="line", x=p[0], y=p[1], x2=q[0], y2=q[1],
                colour=colour, thickness=max(1, int(round(thick)))))

        def circle(p, r, colour):
            state.draw.append(DrawCommand(
                kind="circle", x=p[0], y=p[1], radius=max(1, r),
                colour=colour, thickness=-1))

        def mix(c1, c2, k):
            """Blend two BGR colours, k = 0 gives c1, k = 1 gives c2."""
            return tuple(int(x + (y - x) * k) for x, y in zip(c1, c2))

        base = BAND_COLOURS[pid % len(BAND_COLOURS)]
        black, white = (0, 0, 0), (255, 255, 255)
        shadow = mix(base, black, 0.55)
        dark = mix(base, black, 0.30)
        light = mix(base, white, 0.25)
        shine = mix(base, white, 0.50)

        th = w * 0.12 #Band thickness relative to face width
        band_v, half, sag = -0.56, 0.58, 0.09
        n = 14
        us = [-half + 2 * half * i / n for i in range(n + 1)]

        def centre_v(u):
            return band_v + sag * (u / half) ** 2

        def taper(u):
            # Reduce band thickness towards the sides
            return 1.0 - 0.45 * (abs(u) / half) ** 2

        # Draw the tails with movement based on head velocity and time
        knot = pt(half + 0.01, centre_v(half) + 0.02, depth=-0.15)
        vx = 0.0
        if pid in self.filters:
            vx = float(self.filters[pid].statePost[2, 0])
        swing = max(-0.9, min(0.9, vx / max(w * 4.0, 1.0)))
        segs = 6
        for start_deg, phase, length, tone in ((82, 0.0, 0.75, dark), (94, 1.9, 0.6, shadow)):
            ang0 = a + math.radians(start_deg) + 0.6 * swing
            pts = [knot]
            x, y = knot
            for k in range(segs):
                flutter = 0.10 * (k / segs) * math.sin(2 * math.pi * 1.6 * t + phase + 0.7 * k)
                ang = ang0 + 0.10 * swing * k + flutter
                x += math.cos(ang) * w * length / segs
                y += math.sin(ang) * w * length / segs
                pts.append((x, y))
            # Draw the tail and its lighter crease
            for tone_k, size in ((tone, 0.70), (mix(tone, white, 0.08), 0.22)):
                for k in range(segs):
                    thick = th * size * (1.0 - 0.55 * k / segs)
                    line(pts[k], pts[k + 1], tone_k, thick)

        # Draw a shadow below the band
        for u0, u1 in zip(us, us[1:]):
            um = (u0 + u1) / 2
            off = 0.5 * th * taper(um) / w + 0.012
            line(pt(u0, centre_v(u0) + off), pt(u1, centre_v(u1) + off),
                 mix(shadow, (60, 80, 110), 0.5), max(1, th * 0.12))

        # Draw the band using layered strokes for shading
        layers = ((0.00, 1.00, dark), (-0.12, 0.72, base),
                  (-0.24, 0.42, light), (-0.36, 0.14, shine))
        for off, size, tone in layers:
            for u0, u1 in zip(us, us[1:]):
                um = (u0 + u1) / 2
                tp = taper(um)
                # darker towards the sides, where the band turns away
                side = (abs(um) / half) ** 2
                col = mix(tone, shadow, 0.55 * side)
                dv = off * th * tp / w
                line(pt(u0, centre_v(u0) + dv), pt(u1, centre_v(u1) + dv),
                     col, th * size * tp)

        # Add cloth folds
        fold = mix(base, dark, 0.6)
        for uf, lean in ((-0.30, 0.04), (0.36, -0.04)):
            v = centre_v(uf)
            h = 0.35 * th * taper(uf) / w
            line(pt(uf - lean, v - h), pt(uf + lean, v + h), fold, max(1, th * 0.08))

        # Draw the knot
        kx, ky = knot
        for dx, dy, tone in ((-0.04, -0.02, dark), (0.03, 0.03, shadow), (0.0, 0.0, base)):
            circle((kx + dx * w, ky + dy * w), th * 0.45, tone)
        circle((kx - 0.015 * w, ky - 0.02 * w), th * 0.18, light)

        # the emblem: an off-white cloth patch with a red sun, slightly
        # shaded at the bottom so it sits ON the curved band
        ex, ey = pt(0.0, centre_v(0.0) - 0.01, depth=0.15)
        circle((ex, ey + th * 0.06), th * 0.50, mix((205, 210, 215), black, 0.25))
        circle((ex, ey), th * 0.46, (222, 228, 232))
        circle((ex, ey), th * 0.24, (38, 38, 178))
        circle((ex - th * 0.07, ey - th * 0.08), th * 0.07, (90, 90, 215))

    def reset(self):
        self.filters.clear()
        self.faces.clear()
        self.measured.clear()
        self.missing.clear()
        self.yaw.clear()
        self.hp_shown.clear()
        self._prev_players = []
        self.last_health = {}
        self.last_frame_time = None
