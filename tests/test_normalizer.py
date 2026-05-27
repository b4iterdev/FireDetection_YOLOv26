from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.schema import Detection


def test_normalize_yolo_output():
    # Mock Ultralytics result-like object
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [xyxy]
            self.conf = [conf]
            self.cls = [cls]
            self.data = [[*xyxy, conf, cls]]

    class MockResult:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    boxes = MockBox([10, 20, 110, 120], 0.9, 1)
    result = MockResult(boxes, {0: "smoke", 1: "fire"})

    detections = normalize_yolo_output(result, "cam1", 100, 1622100000.0)
    assert len(detections) == 1
    d = detections[0]
    assert d.class_name == "fire"
    assert d.bbox_area == 10000.0
