import numpy as np

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.falsenet_filter import FalseNetFilter


def make_detection(bbox_area: float = 100.0):
    return Detection(
        source_id="cam1",
        frame_id=1,
        timestamp=100.0,
        class_id=1,
        class_name="fire",
        confidence=0.9,
        bbox_xyxy=[10.0, 10.0, 20.0, 20.0], # Area is 100
        bbox_area=bbox_area,
    )


def test_falsenet_mock_accepts_large_bbox_as_threat():
    """Simulates a large, threatening fire."""
    falsenet_filter = FalseNetFilter(threat_threshold=0.5)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # Bbox area is 10000, frame area is 1,000,000. Ratio is 0.01
    large_detection = make_detection(bbox_area=10000.0) 
    
    decisions = falsenet_filter.classify(frame, [large_detection])
    
    assert len(decisions) == 1
    assert decisions[0].accepted is True
    assert decisions[0].reason == "falsenet_threat"


def test_falsenet_mock_rejects_small_bbox_as_non_threat():
    """Simulates a small, non-threatening fire like a lighter."""
    falsenet_filter = FalseNetFilter(threat_threshold=0.5)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # Bbox area is 100, frame area is 1,000,000. Ratio is 0.0001
    small_detection = make_detection(bbox_area=100.0)
    
    decisions = falsenet_filter.classify(frame, [small_detection])
    
    assert len(decisions) == 1
    assert decisions[0].accepted is False
    assert decisions[0].reason == "falsenet_non_threat"

def test_falsenet_handles_multiple_detections():
    falsenet_filter = FalseNetFilter(threat_threshold=0.5)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    detections = [
        make_detection(bbox_area=10000.0), # Threat
        make_detection(bbox_area=100.0)    # Non-threat
    ]
    
    decisions = falsenet_filter.classify(frame, detections)
    
    assert len(decisions) == 2
    assert decisions[0].accepted is True
    assert decisions[1].accepted is False
