import json
from pathlib import Path

from fire_detection_alarm.filtering.decision import DetectionDecision


class DetectionLogger:
    log_path: Path

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: DetectionDecision) -> None:
        detection = decision.detection
        record = {
            "timestamp": decision.timestamp,
            "source_id": detection.source_id,
            "frame_id": detection.frame_id,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "bbox_xyxy": detection.bbox_xyxy,
            "bbox_area": detection.bbox_area,
            "accepted": decision.accepted,
            "reason": decision.reason,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            _ = log_file.write(json.dumps(record) + "\n")
