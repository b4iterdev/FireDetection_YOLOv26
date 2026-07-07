from collections.abc import Iterable

import cv2
import numpy as np

from fire_detection_alarm.detection.schema import Detection


def render_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    triggered_detections: Iterable[Detection] | None = None,
) -> np.ndarray:
    triggered_ids = {id(detection) for detection in triggered_detections or []}
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        is_triggered = id(d) in triggered_ids
        color = (0, 255, 255) if is_triggered else (0, 0, 255)
        thickness = 3 if is_triggered else 2
        _ = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.class_name} {d.confidence:.2f}"
        if is_triggered:
            label = f"ALARM {label}"
        _ = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
