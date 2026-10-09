
from pathlib import Path
import time

import cv2
import mediapipe as mp

from game_types import Module as Base, DrawCommand


class Module(Base):
    name = "face"

    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "blaze_face_short_range.tflite"
        )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = FaceLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

    def process(self, frame, state):
        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        for landmarks in result.face_landmarks:
            x_coords = [point.x * width for point in landmarks]
            y_coords = [point.y * height for point in landmarks]

            x_min = max(0, min(x_coords))
            y_min = max(0, min(y_coords))
            x_max = min(width - 1, max(x_coords))
            y_max = min(height - 1, max(y_coords))

            state.draw.append(
                DrawCommand(
                    kind="rect",
                    x=x_min,
                    y=y_min,
                    x2=x_max,
                    y2=y_max,
                    colour=(0, 255, 0),
                    thickness=2,
                )
            )

    def reset(self):
        # Preserve monotonically increasing timestamps
        # for the existing VIDEO-mode detector.
        pass
