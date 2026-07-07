import numpy as np

from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.detection.schema import Detection


def make_detection(bbox_xyxy: list[float]) -> Detection:
    return Detection(
        source_id="test",
        frame_id=1,
        timestamp=100.0,
        class_id=1,
        class_name="fire",
        confidence=0.9,
        bbox_xyxy=bbox_xyxy,
        bbox_area=100.0,
    )


def test_render_detections_highlights_triggered_alarm_box():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    normal_detection = make_detection([10.0, 10.0, 30.0, 30.0])
    triggered_detection = make_detection([40.0, 10.0, 60.0, 30.0])

    rendered = render_detections(frame, [normal_detection, triggered_detection], [triggered_detection])

    assert np.array_equal(rendered[10, 10], np.array([0, 0, 255], dtype=np.uint8))
    assert np.array_equal(rendered[10, 40], np.array([0, 255, 255], dtype=np.uint8))
