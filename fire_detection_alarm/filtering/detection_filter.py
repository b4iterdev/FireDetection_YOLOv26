from collections.abc import Sequence

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision


class DetectionFilter:
    allowed_class_ids: Sequence[int]
    min_confidence: float
    min_bbox_area_ratio: float

    def __init__(self, allowed_class_ids: Sequence[int], min_confidence: float, min_bbox_area_ratio: float):
        self.allowed_class_ids = allowed_class_ids
        self.min_confidence = min_confidence
        self.min_bbox_area_ratio = min_bbox_area_ratio

    def check(self, detection: Detection, frame_shape: Sequence[int] | None):
        if frame_shape is None or len(frame_shape) < 2:
            return DetectionDecision(detection, False, "missing_frame_size", detection.timestamp)

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        frame_area = frame_width * frame_height
        if frame_area <= 0:
            return DetectionDecision(detection, False, "missing_frame_size", detection.timestamp)

        if detection.class_id not in self.allowed_class_ids:
            return DetectionDecision(detection, False, "class_not_allowed", detection.timestamp)

        if detection.confidence < self.min_confidence:
            return DetectionDecision(detection, False, "low_confidence", detection.timestamp)

        bbox_area_ratio: float = detection.bbox_area / frame_area
        if bbox_area_ratio < self.min_bbox_area_ratio:
            return DetectionDecision(detection, False, "bbox_too_small", detection.timestamp)

        return DetectionDecision(detection, True, "accepted", detection.timestamp)
