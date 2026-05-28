from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.detection_filter import DetectionFilter


def make_detection(class_id: int = 1, confidence: float = 0.9, bbox_area: float = 10000.0):
    return Detection(
        source_id="cam1",
        frame_id=1,
        timestamp=100.0,
        class_id=class_id,
        class_name="fire",
        confidence=confidence,
        bbox_xyxy=[10.0, 10.0, 110.0, 110.0],
        bbox_area=bbox_area,
    )


def test_detection_filter_accepts_valid_detection():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )
    decision = detection_filter.check(make_detection(), frame_shape=(1000, 1000, 3))
    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.detection.class_id == 1


def test_detection_filter_rejects_disallowed_class():
    detection_filter = DetectionFilter([1], 0.65, 0.002)
    decision = detection_filter.check(make_detection(class_id=2), frame_shape=(1000, 1000, 3))
    assert decision.accepted is False
    assert decision.reason == "class_not_allowed"


def test_detection_filter_rejects_low_confidence():
    detection_filter = DetectionFilter([1], 0.65, 0.002)
    decision = detection_filter.check(make_detection(confidence=0.5), frame_shape=(1000, 1000, 3))
    assert decision.accepted is False
    assert decision.reason == "low_confidence"


def test_detection_filter_rejects_small_bbox_ratio():
    detection_filter = DetectionFilter([1], 0.65, 0.002)
    decision = detection_filter.check(make_detection(bbox_area=100.0), frame_shape=(1000, 1000, 3))
    assert decision.accepted is False
    assert decision.reason == "bbox_too_small"


def test_detection_filter_rejects_missing_frame_shape():
    detection_filter = DetectionFilter([1], 0.65, 0.002)
    decision = detection_filter.check(make_detection(), frame_shape=None)
    assert decision.accepted is False
    assert decision.reason == "missing_frame_size"
