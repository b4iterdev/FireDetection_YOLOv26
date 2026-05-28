from dataclasses import dataclass

from fire_detection_alarm.detection.schema import Detection


@dataclass
class DetectionDecision:
    detection: Detection
    accepted: bool
    reason: str
    timestamp: float
