from pathlib import Path

import cv2
import numpy as np
import pytest

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.behavior_tracker import BehaviorTracker
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.detection_filter import DetectionFilter
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.inputs.video_source import DEFAULT_VIDEO_FPS
from fire_detection_alarm.models.yolo_engine import YOLOEngine
from fire_detection_alarm.web import pipeline
from fire_detection_alarm.web.pipeline import WebPipelineRunner


def _config() -> dict[str, dict[str, object]]:
    return {
        "model": {"device": "cpu", "image_size": 640},
        "classes": {"allowed": [0, 1]},
        "inference": {"confidence_threshold": 0.25, "iou_threshold": 0.45},
        "filtering": {
            "min_bbox_area_ratio": 0.0,
            "min_persistence_seconds": 0.0,
            "min_consecutive_frames": 1,
        },
        "behavior_tracking": {
            "min_track_frames": 1,
            "max_stable_growth_ratio": 1.0,
            "max_non_hazard_area_ratio": 1.0,
            "min_growth_ratio": 0.0,
        },
    }


class FakeEngine:
    def __init__(self, model_path: str, device: str):
        self.model_path: str = model_path
        self.device: str = device


class NoDecisionRunner(WebPipelineRunner):
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        source_id: str,
        engine: YOLOEngine,
        cfg: object,
        detection_filter: DetectionFilter,
        behavior_tracker: BehaviorTracker,
        temporal_filter: TemporalFilter,
    ) -> list[DetectionDecision]:
        return []


class RaisingRunner(NoDecisionRunner):
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        source_id: str,
        engine: YOLOEngine,
        cfg: object,
        detection_filter: DetectionFilter,
        behavior_tracker: BehaviorTracker,
        temporal_filter: TemporalFilter,
    ) -> list[DetectionDecision]:
        if frame_id == 1:
            raise RuntimeError("boom")
        return []


class FakeVideoWriter:
    instances: list["FakeVideoWriter"] = []

    def __init__(self, path: str, fourcc: int, fps: float, frame_size: tuple[int, int]):
        self.path: str = path
        self.codec: int = fourcc
        self.fps: float = fps
        self.frame_size: tuple[int, int] = frame_size
        self.frames: list[np.ndarray] = []
        self.released: bool = False
        FakeVideoWriter.instances.append(self)

    @staticmethod
    def fourcc(*codes: str) -> int:
        assert codes == ("a", "v", "c", "1")
        return 1234

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class FakeVideoSource:
    instances: list["FakeVideoSource"] = []
    frames: list[np.ndarray] = [
        np.full((4, 6, 3), 1, dtype=np.uint8),
        np.full((4, 6, 3), 2, dtype=np.uint8),
        np.full((4, 6, 3), 3, dtype=np.uint8),
    ]

    def __init__(self, path: str):
        self.path: str = path
        self.index: int = 0
        self.released: bool = False
        self.write_counts_at_read: list[int | None] = []
        FakeVideoSource.instances.append(self)

    def read(self) -> tuple[bool, np.ndarray | None]:
        writer = FakeVideoWriter.instances[0] if FakeVideoWriter.instances else None
        self.write_counts_at_read.append(len(writer.frames) if writer is not None else None)
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def fps(self) -> float:
        return 12.5

    def release(self) -> None:
        self.released = True


class EmptyVideoSource(FakeVideoSource):
    frames: list[np.ndarray] = []


class FallbackFpsVideoSource:
    instances: list["FallbackFpsVideoSource"] = []
    frames: list[np.ndarray] = FakeVideoSource.frames

    def __init__(self, path: str):
        self.path: str = path
        self.index: int = 0
        self.released: bool = False
        FallbackFpsVideoSource.instances.append(self)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def fps(self) -> float:
        return DEFAULT_VIDEO_FPS

    def release(self) -> None:
        self.released = True


class FpsErrorVideoSource:
    instances: list["FpsErrorVideoSource"] = []

    def __init__(self, path: str):
        self.path: str = path
        self.released: bool = False
        FpsErrorVideoSource.instances.append(self)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def fps(self) -> float:
        raise RuntimeError("fps unavailable")

    def release(self) -> None:
        self.released = True


class FakeImageSource:
    instances: list["FakeImageSource"] = []

    def __init__(self, path: str):
        self.path: str = path
        self.frame: np.ndarray | None = np.full((5, 7, 3), 9, dtype=np.uint8)
        self.released: bool = False
        FakeImageSource.instances.append(self)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frame is None:
            return False, None
        frame = self.frame
        self.frame = None
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def fake_render_detections(
    frame: np.ndarray,
    _all_detections: list[Detection],
    _accepted_detections: list[Detection],
) -> np.ndarray:
    return frame + 10


def fake_imwrite(writes: list[tuple[str, np.ndarray]], path: str, frame: np.ndarray) -> bool:
    writes.append((path, frame.copy()))
    return True


@pytest.fixture(autouse=True)
def patch_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeVideoWriter.instances = []
    FakeVideoSource.instances = []
    FallbackFpsVideoSource.instances = []
    FpsErrorVideoSource.instances = []
    FakeImageSource.instances = []
    monkeypatch.setattr(pipeline, "load_config", _config)
    monkeypatch.setattr(pipeline, "YOLOEngine", FakeEngine)
    monkeypatch.setattr(cv2, "VideoWriter", FakeVideoWriter)
    monkeypatch.setattr(pipeline, "render_detections", fake_render_detections)


def test_video_pipeline_streams_frames_with_source_fps_and_releases_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "VideoSource", FakeVideoSource)
    runner = NoDecisionRunner(tmp_path / "results", tmp_path / "logs")

    result = runner.run(tmp_path / "sample.mp4", "model.pt")

    writer = FakeVideoWriter.instances[0]
    source = FakeVideoSource.instances[0]
    assert result.output_path == tmp_path / "results" / "sample_annotated.mp4"
    assert result.frame_count == 3
    assert writer.path == str(result.output_path)
    assert writer.fps == 12.5
    assert writer.frame_size == (6, 4)
    assert len(writer.frames) == 3
    assert source.write_counts_at_read == [None, 1, 2, 3]
    assert writer.released is True
    assert source.released is True


def test_video_pipeline_releases_writer_and_source_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "VideoSource", FakeVideoSource)
    runner = RaisingRunner(tmp_path / "results", tmp_path / "logs")

    with pytest.raises(RuntimeError, match="boom"):
        _ = runner.run(tmp_path / "sample.mp4", "model.pt")

    assert FakeVideoWriter.instances[0].released is True
    assert FakeVideoSource.instances[0].released is True


def test_video_pipeline_uses_fallback_fps_from_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "VideoSource", FallbackFpsVideoSource)
    runner = NoDecisionRunner(tmp_path / "results", tmp_path / "logs")

    result = runner.run(tmp_path / "sample.mp4", "model.pt")

    assert result.frame_count == 3
    assert FakeVideoWriter.instances[0].fps == DEFAULT_VIDEO_FPS


def test_video_pipeline_empty_input_returns_expected_path_without_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "VideoSource", EmptyVideoSource)
    runner = NoDecisionRunner(tmp_path / "results", tmp_path / "logs")

    result = runner.run(tmp_path / "empty.mp4", "model.pt")

    assert result.output_path == tmp_path / "results" / "empty_annotated.mp4"
    assert result.output_available is False
    assert result.frame_count == 0
    assert FakeVideoWriter.instances == []
    assert EmptyVideoSource.instances[0].released is True


def test_video_pipeline_releases_source_when_fps_lookup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "VideoSource", FpsErrorVideoSource)
    runner = NoDecisionRunner(tmp_path / "results", tmp_path / "logs")

    with pytest.raises(RuntimeError, match="fps unavailable"):
        _ = runner.run(tmp_path / "sample.mp4", "model.pt")

    assert FakeVideoWriter.instances == []
    assert FpsErrorVideoSource.instances[0].released is True


def test_image_pipeline_writes_single_annotated_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, np.ndarray]] = []

    def capture_imwrite(path: str, frame: np.ndarray) -> bool:
        return fake_imwrite(writes, path, frame)

    monkeypatch.setattr(pipeline, "ImageSource", FakeImageSource)
    monkeypatch.setattr(cv2, "imwrite", capture_imwrite)
    runner = NoDecisionRunner(tmp_path / "results", tmp_path / "logs")

    result = runner.run(tmp_path / "sample.jpg", "model.pt")

    assert result.output_path == tmp_path / "results" / "sample_annotated.jpg"
    assert result.frame_count == 1
    assert len(writes) == 1
    assert writes[0][0] == str(result.output_path)
    assert np.array_equal(writes[0][1], np.full((5, 7, 3), 19, dtype=np.uint8))
    assert FakeImageSource.instances[0].released is True
