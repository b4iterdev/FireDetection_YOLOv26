import pytest
import numpy as np
import cv2
import math
from pathlib import Path
from fire_detection_alarm.inputs.image_source import ImageSource
from fire_detection_alarm.inputs.video_source import DEFAULT_VIDEO_FPS, VideoSource

def test_image_source(tmp_path: Path):
    img_path = str(tmp_path / "test.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = cv2.imwrite(img_path, img)
    
    source = ImageSource(img_path)
    ret, frame = source.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (100, 100, 3)
    
    ret, frame = source.read()
    assert ret is False
    assert frame is None

def test_video_source_release():
    source = VideoSource("dummy_path.mp4")
    source.release()


def test_video_source_fps_returns_capture_fps(monkeypatch: pytest.MonkeyPatch):
    class FakeCapture:
        def read(self) -> tuple[bool, None]:
            return False, None

        def get(self, prop_id: int) -> float:
            assert prop_id == cv2.CAP_PROP_FPS
            return 29.97

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())

    source = VideoSource("sample.mp4")

    assert source.fps() == 29.97


@pytest.mark.parametrize("capture_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_video_source_fps_falls_back_for_invalid_values(monkeypatch: pytest.MonkeyPatch, capture_fps: float):
    class FakeCapture:
        def read(self) -> tuple[bool, None]:
            return False, None

        def get(self, prop_id: int) -> float:
            assert prop_id == cv2.CAP_PROP_FPS
            return capture_fps

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())

    source = VideoSource("sample.mp4")

    assert source.fps() == 20.0


def test_video_source_media_time_uses_frame_id_over_valid_fps_without_position_msec(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeCapture:
        def read(self) -> tuple[bool, None]:
            return False, None

        def get(self, prop_id: int) -> float:
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                raise AssertionError("media time must not read CAP_PROP_POS_MSEC")
            if prop_id == cv2.CAP_PROP_FPS:
                return 25.0
            raise AssertionError(f"unexpected property {prop_id}")

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())

    source = VideoSource("sample.mp4")

    assert source.media_time_seconds(frame_id=5) == 0.2


@pytest.mark.parametrize("capture_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_video_source_media_time_uses_default_fps_fallback_without_position_msec(
    monkeypatch: pytest.MonkeyPatch,
    capture_fps: float,
):
    class FakeCapture:
        def read(self) -> tuple[bool, None]:
            return False, None

        def get(self, prop_id: int) -> float:
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                raise AssertionError("media time must not read CAP_PROP_POS_MSEC")
            if prop_id == cv2.CAP_PROP_FPS:
                return capture_fps
            raise AssertionError(f"unexpected property {prop_id}")

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())

    source = VideoSource("sample.mp4")

    assert math.isclose(source.media_time_seconds(frame_id=3), 3.0 / DEFAULT_VIDEO_FPS)
