from __future__ import annotations

from pathlib import Path
import time

import cv2
import mediapipe as mp


BODY_CONNECTIONS = (
    (11, 12),  # shoulders
    (11, 13),  # left shoulder → left elbow
    (13, 15),  # left elbow → left wrist
    (12, 14),  # right shoulder → right elbow
    (14, 16),  # right elbow → right wrist
    (11, 23),  # left shoulder → left hip
    (12, 24),  # right shoulder → right hip
    (23, 24),  # hips
    (23, 25),  # left hip → left knee
    (25, 27),  # left knee → left ankle
    (27, 29),  # left ankle → left heel
    (29, 31),  # left heel → left foot
    (24, 26),  # right hip → right knee
    (26, 28),  # right knee → right ankle
    (28, 30),  # right ankle → right heel
    (30, 32),  # right heel → right foot
    (15, 17),  # left wrist → pinky anchor
    (15, 19),  # left wrist → index anchor
    (15, 21),  # left wrist → thumb anchor
    (16, 18),  # right wrist → pinky anchor
    (16, 20),  # right wrist → index anchor
    (16, 22),  # right wrist → thumb anchor
)


class Module:
    name = "pose"

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "pose_landmarker_full.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )
        self.last_timestamp_ms = 0
        self.last_result = None
        self.pose_found = False

    def process(self, frame, state) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = max(
            self.last_timestamp_ms + 1,
            int(state.timestamp * 1000),
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        self.last_result = result
        self.pose_found = bool(result.pose_landmarks)

        if self.pose_found:
            self._draw_debug_skeleton(frame, result.pose_landmarks[0])

    def reset(self) -> None:
        self.pose_found = False
        self.last_result = None
        self.last_timestamp_ms = 0

    def _draw_debug_skeleton(self, frame, landmarks) -> None:
        height, width = frame.shape[:2]

        pixel_points = []
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            pixel_points.append((x, y))

        for start_index, end_index in BODY_CONNECTIONS:
            start = landmarks[start_index]
            end = landmarks[end_index]

            if start.visibility < 0.5 or end.visibility < 0.5:
                continue

            cv2.line(
                frame,
                pixel_points[start_index],
                pixel_points[end_index],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for index, landmark in enumerate(landmarks):
            if landmark.visibility < 0.5:
                continue

            cv2.circle(
                frame,
                pixel_points[index],
                4,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
