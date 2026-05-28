import json
from pathlib import Path

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.logging.detection_logger import DetectionLogger


def test_detection_logger_writes_jsonl_record(tmp_path: Path) -> None:
    log_path = tmp_path / "detections.jsonl"
    logger = DetectionLogger(str(log_path))
    detection = Detection(
        source_id="cam1",
        frame_id=7,
        timestamp=123.4,
        class_id=1,
        class_name="fire",
        confidence=0.91,
        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        bbox_area=4.0,
    )
    decision = DetectionDecision(
        detection=detection,
        accepted=True,
        reason="accepted",
        timestamp=123.5,
    )

    logger.write(decision)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records == [
        {
            "timestamp": 123.5,
            "source_id": "cam1",
            "frame_id": 7,
            "class_id": 1,
            "class_name": "fire",
            "confidence": 0.91,
            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            "bbox_area": 4.0,
            "accepted": True,
            "reason": "accepted",
        }
    ]
