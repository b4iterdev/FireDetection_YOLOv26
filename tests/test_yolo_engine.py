from unittest.mock import MagicMock, patch

from fire_detection_alarm.models.yolo_engine import YOLOEngine


@patch("fire_detection_alarm.models.yolo_engine.YOLO")
def test_engine_predict(mock_yolo):
    engine = YOLOEngine("dummy.pt")
    mock_model = mock_yolo.return_value
    mock_model.predict.return_value = [MagicMock(boxes=MagicMock(data=[]), names={0: "smoke"})]

    results = engine.predict("frame.jpg")
    assert len(results) == 1
    mock_model.predict.assert_called_once()
