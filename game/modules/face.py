
from pathlib import Path
import time
import math

import cv2
import mediapipe as mp

from game_types import Module as Base, DrawCommand


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

    def process(self, frame, state):
        height, width = frame.shape[:2]

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

        # BlazeFace returns detections, not face_landmarks
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

            # BlazeFace keypoints:
            # 0 = right eye, 1 = left eye
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

            # Draw face bounding box
            state.draw.append(
                DrawCommand(
                    kind="rect",
                    x=x_min,
                    y=y_min,
                    x2=x_max,
                    y2=y_max,
                    colour=(0, 255, 0),
                    thickness=2
                )
            )

    def reset(self):
        # No persistent face tracking state yet
        pass

