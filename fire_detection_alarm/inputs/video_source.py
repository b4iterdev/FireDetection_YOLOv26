import math

import cv2
import numpy as np
from fire_detection_alarm.inputs.base import BaseSource

DEFAULT_VIDEO_FPS = 20.0

class VideoSource(BaseSource):
    def __init__(self, path: str):
        self.cap: cv2.VideoCapture = cv2.VideoCapture(path)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def fps(self) -> float:
        capture_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if math.isfinite(capture_fps) and capture_fps > 0:
            return capture_fps
        return DEFAULT_VIDEO_FPS

    def media_time_seconds(self, frame_id: int) -> float:
        return float(frame_id) / self.fps()

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
