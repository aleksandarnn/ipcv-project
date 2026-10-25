
from pathlib import Path
import time
import math

import cv2
import mediapipe as mp

from game_types import Module as Base, DrawCommand, Face


class Module(Base):
    name = "face"

    def __init__(self):
        # Locate the BlazeFace model
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "blaze_face_full_range.tflite"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"BlazeFace model not found: {model_path}"
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=0.5
        )

        self.detector = FaceDetector.create_from_options(options)

        # Timestamps must increase in VIDEO mode
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

        # EMA smoothing parameters
        self.alpha = 0.5
        self.smoothed_faces = {}

    def process(self, frame, state):
        height, width = frame.shape[:2]

        for player in state.players:
            player.face = None

        # OpenCV uses BGR; MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        # Detect faces
        result = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        for detection in result.detections:
            bbox = detection.bounding_box

            x_min = max(0, bbox.origin_x)
            y_min = max(0, bbox.origin_y)

            x_max = min(
                width - 1,
                bbox.origin_x + bbox.width
            )
            y_max = min(
                height - 1,
                bbox.origin_y + bbox.height
            )

            # Face geometry
            face_cx = (x_min + x_max) / 2
            face_cy = (y_min + y_max) / 2

            face_width = x_max - x_min
            face_height = y_max - y_min

            # Estimate face rotation using eye keypoints
            keypoints = detection.keypoints
            face_angle = 0.0

            if len(keypoints) >= 2:
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                dx = (left_eye.x - right_eye.x) * width
                dy = (left_eye.y - right_eye.y) * height

                face_angle = math.degrees(
                    math.atan2(dy, dx)
                )

            # Create face information
            face = Face(
                cx=face_cx,
                cy=face_cy,
                w=face_width,
                h=face_height,
                angle=face_angle,
                confidence=detection.categories[0].score
            )

            matched_player = None
            best_distance = float("inf")

            # Find the player corresponding to this face
            for player in state.players:

                if player.bbox is None or not player.visible:
                    continue

                # Avoid assigning multiple faces to the same player
                if player.face is not None:
                    continue

                box = player.bbox

                # Check whether face centre lies inside player bbox
                if box.x0 <= face_cx <= box.x1 and box.y0 <= face_cy <= box.y1:

                    # Estimate upper-body/head position
                    target_x = (box.x0 + box.x1) / 2
                    target_y = box.y0 + box.h * 0.2

                    distance = (
                        (face_cx - target_x) ** 2
                        + (face_cy - target_y) ** 2
                    )

                    if distance < best_distance:
                        best_distance = distance
                        matched_player = player

            # Store smoothed face information
            if matched_player is not None:
                pid = matched_player.player_id

                if pid in self.smoothed_faces:
                    prev = self.smoothed_faces[pid]
                    a = self.alpha

                    # EMA smoothing
                    face = Face(
                        cx=a * face.cx + (1 - a) * prev.cx,
                        cy=a * face.cy + (1 - a) * prev.cy,
                        w=a * face.w + (1 - a) * prev.w,
                        h=a * face.h + (1 - a) * prev.h,
                        angle=a * face.angle + (1 - a) * prev.angle,
                        confidence=face.confidence
                    )

                self.smoothed_faces[pid] = face
                matched_player.face = face

                # Use smoothed face geometry for drawing
                x_min = face.cx - face.w / 2
                y_min = face.cy - face.h / 2
                x_max = face.cx + face.w / 2
                y_max = face.cy + face.h / 2

            # Draw face bounding box
            state.draw.append(
                DrawCommand(
                    kind="rect",
                    x=int(x_min),
                    y=int(y_min),
                    x2=int(x_max),
                    y2=int(y_max),
                    colour=(0, 255, 0),
                    thickness=2
                )
            )

    def reset(self):
        self.smoothed_faces.clear()

