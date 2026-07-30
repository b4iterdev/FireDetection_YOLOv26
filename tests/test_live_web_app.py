import io
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.web.app import create_app
from fire_detection_alarm.web import live as live_module
from fire_detection_alarm.web.live import LiveDetectionSession


class FakeLiveSession:
    def __init__(self):
        self.started_payload: dict[str, object] | None = None
        self.stopped: bool = False
        self.processed_frame: bytes | None = None
        self.cfg: dict[str, dict[str, object]] = {
            "model": {"image_size": 640},
            "inference": {"max_fps": 5},
        }

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self) -> dict[str, object]:
        self.stopped = True
        return {"running": False}

    def status(self) -> dict[str, object]:
        summary = self.result()
        running = self.started_payload is not None and not self.stopped
        return {
            "running": running,
            "source_type": self.started_payload.get("source_type", "") if self.started_payload is not None else "",
            "frame_count": 0,
            "summary_available": summary is not None,
            "summary_url": "/live/result" if summary is not None else "",
            "completed_reason": summary.completed_reason if summary is not None else "",
        }

    def result(self):
        return getattr(self, "summary", None)

    def mjpeg_frames(self) -> Iterator[bytes]:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"

    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        self.processed_frame = encoded_frame
        return _jpeg_bytes()


class RaisingFrameLiveSession(FakeLiveSession):
    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        raise ValueError("invalid JPEG frame")


class FakeEngine:
    def __init__(self, model_path: str, device: str):
        self.model_path: str = model_path
        self.device: str = device


class PredictingFakeEngine(FakeEngine):
    def predict(self, frame: np.ndarray, conf: float, iou: float, imgsz: int) -> list[SimpleNamespace]:
        _ = (frame, conf, iou, imgsz)
        boxes = SimpleNamespace(data=np.array([[0.0, 0.0, 4.0, 4.0, 0.9, 1.0]]))
        return [SimpleNamespace(boxes=boxes, names={0: "smoke", 1: "fire"})]


def _config() -> dict[str, dict[str, object]]:
    return {
        "model": {"path": "model.pt", "device": "cpu", "image_size": 640},
        "classes": {"allowed": [0, 1]},
        "inference": {"confidence_threshold": 0.25, "iou_threshold": 0.45, "max_fps": 5},
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


def _persistent_config() -> dict[str, dict[str, object]]:
    config = _config()
    config["filtering"] = {
        "min_bbox_area_ratio": 0.0,
        "min_persistence_seconds": 2.0,
        "min_consecutive_frames": 3,
    }
    return config


def _jpeg_bytes() -> bytes:
    frame = np.full((4, 6, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_live_page_renders_controls(tmp_path: Path) -> None:
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/live")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Live Fire Monitor" in body
    assert "Dashboard" not in body
    assert "Firewatch" not in body
    assert "Unified analysis workspace" not in body
    assert "Analyze a webcam, uploaded video, or RTSP source in one place." not in body
    assert "Stop at any time to review the session summary." not in body
    assert "<header" not in body
    assert "rtsp_url" in body
    assert "Frames analyzed" in body
    assert "Accepted alarms" in body
    assert "Raw session diagnostics" in body
    assert 'aria-live="polite"' in body
    assert "Upload video for live detection" not in body
    assert 'id="live-video-file"' in body
    assert 'type="file"' in body
    assert "navigator.mediaDevices.getUserMedia" in body
    assert "/api/live/start" in body
    assert "/api/live/status" in body
    assert "/api/live/stop" in body
    assert "/api/live/upload" in body
    assert "/api/live/frame" in body
    assert "/stream.mjpeg" in body
    assert "videoSummaryArmed" in body
    assert "summaryNavigationStarted" in body
    assert "navigateToSummary(data)" in body
    assert 'id="stream-placeholder"' in body
    assert "showLivePlayer()" in body
    assert "clearLivePlayer()" in body
    assert "if (showSummary) navigateToSummary(data)" in body
    assert "stopButton.addEventListener('click', () => stopLive(true))" in body
    assert "sourceType.addEventListener('change'" in body
    assert "await stopLive(false)" in body
    assert 'id="camera_index"' not in body
    assert 'id="file_path"' not in body


def test_live_page_receives_load_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_context: dict[str, object] = {}
    config = _config()
    config["model"]["image_size"] = 416
    config["inference"]["max_fps"] = 12

    def fake_render_template(template: str, **context: object) -> str:
        captured_context.update(context)
        return template

    monkeypatch.setattr("fire_detection_alarm.web.app.load_config", lambda: config)
    monkeypatch.setattr("flask.render_template", fake_render_template)
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/live")

    assert response.status_code == 200
    assert captured_context["max_fps"] == 12
    assert captured_context["image_size"] == 416


def test_live_start_accepts_webcam_and_rtsp(tmp_path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    for payload in (
        {"source_type": "webcam"},
        {"source_type": "rtsp", "rtsp_url": "rtsp://example.local/stream"},
    ):
        response = client.post("/api/live/start", json=payload)
        assert response.status_code == 200
        assert response.get_json()["running"] is True


def test_live_start_rejects_direct_server_video_path(tmp_path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    response = client.post(
        "/api/live/start",
        json={"source_type": "video_file", "file_path": "/tmp/video.mp4"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "source_type must be webcam or rtsp"}
    assert live_session.started_payload is None


def test_live_start_rejects_invalid_source_type(tmp_path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.post("/api/live/start", json={"source_type": "bad"})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_live_stop_and_status_routes(tmp_path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    client.post("/api/live/start", json={"source_type": "webcam", "camera_index": 0})
    status_response = client.get("/api/live/status")
    stop_response = client.post("/api/live/stop")

    assert status_response.status_code == 200
    assert status_response.get_json()["running"] is True
    assert stop_response.status_code == 200
    assert stop_response.get_json()["running"] is False


def test_live_stream_route_returns_mjpeg_for_running_server_source(tmp_path):
    live_session = FakeLiveSession()
    live_session.start({"source_type": "rtsp", "rtsp_url": "rtsp://example.local/stream"})
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    response = client.get("/stream.mjpeg")

    assert response.status_code == 200
    assert response.content_type.startswith("multipart/x-mixed-replace")


def test_live_stream_route_returns_json_409_before_start(tmp_path: Path) -> None:
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/stream.mjpeg")

    assert response.status_code == 409
    assert response.get_json() == {"error": "live stream is not available"}


def test_live_stream_route_returns_json_409_for_webcam_session(tmp_path: Path) -> None:
    live_session = FakeLiveSession()
    live_session.start({"source_type": "webcam"})
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    response = client.get("/stream.mjpeg")

    assert response.status_code == 409
    assert response.get_json() == {"error": "live stream is not available"}


def test_live_video_upload_saves_file_and_starts_session_with_resolved_path(tmp_path: Path):
    live_session = FakeLiveSession()
    upload_dir = tmp_path / "uploads"
    app = create_app(upload_dir=upload_dir, result_dir=tmp_path / "results", live_session=live_session)
    client = app.test_client()

    response = client.post(
        "/api/live/upload",
        data={"file": (io.BytesIO(b"video"), "../sample.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["running"] is True
    assert live_session.started_payload is not None
    assert live_session.started_payload["source_type"] == "video_file"
    saved_path = Path(str(live_session.started_payload["file_path"]))
    assert saved_path.is_absolute()
    assert saved_path.parent == upload_dir.resolve()
    assert saved_path.name.endswith("_sample.mp4")
    assert saved_path.read_bytes() == b"video"


@pytest.mark.parametrize(
    ("data", "expected_error"),
    [
        ({}, "Choose a video file."),
        ({"file": (io.BytesIO(b"video"), "sample.txt")}, "Upload a supported video file."),
        ({"file": (io.BytesIO(b"video"), "")}, "Choose a video file."),
    ],
)
def test_live_video_upload_rejects_missing_or_unsupported_file(
    tmp_path: Path,
    data: dict[str, object],
    expected_error: str,
):
    live_session = FakeLiveSession()
    upload_dir = tmp_path / "uploads"
    app = create_app(upload_dir=upload_dir, result_dir=tmp_path / "results", live_session=live_session)
    client = app.test_client()

    response = client.post("/api/live/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == expected_error
    assert live_session.started_payload is None
    assert list(upload_dir.iterdir()) == []


def test_live_video_upload_rejects_start_errors(tmp_path: Path):
    class StartErrorLiveSession(FakeLiveSession):
        def start(self, payload: dict[str, object]) -> dict[str, object]:
            raise ValueError("cannot start")

    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=StartErrorLiveSession(),
    )
    client = app.test_client()

    response = client.post(
        "/api/live/upload",
        data={"file": (io.BytesIO(b"video"), "sample.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "cannot start"


def test_live_frame_endpoint_returns_annotated_jpeg(tmp_path: Path):
    live_session = FakeLiveSession()
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results", live_session=live_session)
    client = app.test_client()
    payload = _jpeg_bytes()

    response = client.post("/api/live/frame", data=payload, content_type="image/jpeg")

    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert response.data.startswith(b"\xff\xd8")
    assert live_session.processed_frame == payload


def test_live_frame_endpoint_maps_value_error_to_json_400(tmp_path: Path):
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=RaisingFrameLiveSession(),
    )
    client = app.test_client()

    response = client.post("/api/live/frame", data=b"bad", content_type="image/jpeg")

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid JPEG frame"}


def test_webcam_start_initializes_without_opening_server_camera(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_video_capture(source: object) -> object:
        raise AssertionError(f"VideoCapture should not be opened for browser webcam: {source}")

    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    monkeypatch.setattr(cv2, "VideoCapture", raise_video_capture)
    session = LiveDetectionSession(tmp_path / "results")

    status = session.start({"source_type": "webcam", "camera_index": 9})

    assert status["running"] is True
    assert status["source_type"] == "webcam"
    assert session.capture is None
    assert isinstance(session.engine, FakeEngine)


def test_browser_frame_processing_decodes_processes_and_returns_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    processed_shapes: list[tuple[int, ...]] = []

    def fake_process_frame(frame: np.ndarray) -> np.ndarray:
        processed_shapes.append(frame.shape)
        session.status_state.frame_count += 1
        return np.full(frame.shape, 255, dtype=np.uint8)

    monkeypatch.setattr(session, "_process_frame", fake_process_frame)
    session.start({"source_type": "webcam"})

    result = session.process_browser_frame(_jpeg_bytes())

    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert result.startswith(b"\xff\xd8")
    assert decoded is not None
    assert processed_shapes == [(4, 6, 3)]
    assert session.status_state.frame_count == 1


@pytest.mark.parametrize("payload", [b"", b"not a jpeg"])
def test_browser_frame_processing_rejects_empty_or_invalid_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})

    with pytest.raises(ValueError, match="invalid JPEG frame"):
        session.process_browser_frame(payload)


def test_live_uploaded_video_file_persistence_uses_media_time_while_records_keep_wall_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions: list[DetectionDecision] = []

    class FakeVideoCapture:
        def __init__(self, source: str) -> None:
            self.source: str = source
            self.frames: list[np.ndarray] = [
                np.full((4, 6, 3), 1, dtype=np.uint8),
                np.full((4, 6, 3), 2, dtype=np.uint8),
                np.full((4, 6, 3), 3, dtype=np.uint8),
            ]
            self.released: bool = False

        def read(self) -> tuple[bool, np.ndarray | None]:
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def get(self, prop_id: int) -> float:
            if prop_id == cv2.CAP_PROP_FPS:
                return 0.9
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 6.0
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 4.0
            raise AssertionError(f"unexpected capture property {prop_id}")

        def release(self) -> None:
            self.released = True

    def capture_decision(
        detection,
        accepted: bool,
        reason: str,
        timestamp: float,
    ) -> DetectionDecision:
        decision = DetectionDecision(detection, accepted, reason, timestamp)
        decisions.append(decision)
        return decision

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    wall_times = iter([1000.0, 1000.1, 1000.2])
    monkeypatch.setattr(live_module, "load_config", _persistent_config)
    monkeypatch.setattr(live_module, "YOLOEngine", PredictingFakeEngine)
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(live_module, "DetectionDecision", capture_decision)
    monkeypatch.setattr("fire_detection_alarm.web.live.time.time", lambda: next(wall_times))
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(cv2, "imwrite", lambda path, frame: True)
    session = LiveDetectionSession(tmp_path / "results")

    status = session.start({"source_type": "video_file", "file_path": str(video_path)})
    for _ in range(3):
        assert session.capture is not None
        ret, frame = session.capture.read()
        assert ret is True
        assert frame is not None
        _ = session._process_frame(frame)

    assert status["source_type"] == "video_file"
    assert [decision.reason for decision in decisions] == [
        "not_persistent",
        "not_persistent",
        "accepted",
    ]
    assert [decision.timestamp for decision in decisions] == [1000.0, 1000.1, 1000.2]
    assert [decision.detection.timestamp for decision in decisions] == [1000.0, 1000.1, 1000.2]
    assert session.status_state.accepted_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"source_type": "webcam"},
        {"source_type": "rtsp", "rtsp_url": "rtsp://example.local/stream"},
    ],
)
def test_live_webcam_and_rtsp_persistence_remain_wall_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    class FakeRtspCapture:
        def __init__(self, source: str) -> None:
            self.source: str = source

        def release(self) -> None:
            pass

    class SpyTemporalFilter(TemporalFilter):
        def __init__(self) -> None:
            super().__init__(min_seconds=2.0, min_frames=3)
            self.timestamps: list[float] = []

        def check(self, source_id: str, is_detected: bool, timestamp: float) -> bool:
            _ = (source_id, is_detected)
            self.timestamps.append(timestamp)
            return False

    monkeypatch.setattr(live_module, "load_config", _persistent_config)
    monkeypatch.setattr(live_module, "YOLOEngine", PredictingFakeEngine)
    monkeypatch.setattr(cv2, "VideoCapture", FakeRtspCapture)
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    wall_times = iter([2000.0, 2000.1, 2000.2])
    monkeypatch.setattr("fire_detection_alarm.web.live.time.time", lambda: next(wall_times))
    session = LiveDetectionSession(tmp_path / "results")

    _ = session.start(payload)
    spy_temporal_filter = SpyTemporalFilter()
    session.temporal_filter = spy_temporal_filter
    for _ in range(3):
        _ = session._process_frame(np.full((4, 6, 3), 1, dtype=np.uint8))

    assert spy_temporal_filter.timestamps == [2000.0, 2000.1, 2000.2]


def test_mjpeg_stream_respects_configured_max_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = [frame, frame]
            self.released: bool = False

        def read(self) -> tuple[bool, np.ndarray | None]:
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    session = LiveDetectionSession(tmp_path / "results")
    session.cfg = {"inference": {"max_fps": 5}}
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: FakeCapture())
    session.capture = cv2.VideoCapture("fake")
    session.status_state.running = True
    monkeypatch.setattr(session, "_process_frame", lambda current_frame: current_frame)
    monotonic_values = iter([0.0, 0.0, 0.05, 0.2])
    sleep_calls: list[float] = []
    monkeypatch.setattr(live_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(live_module.time, "sleep", sleep_calls.append)

    chunks = list(session.mjpeg_frames())

    assert len(chunks) == 2
    assert sleep_calls == pytest.approx([0.15])


class FakeLiveVideoWriter:
    instances: list["FakeLiveVideoWriter"] = []

    def __init__(self, path: str, fourcc: int, fps: float, frame_size: tuple[int, int]):
        self.path: str = path
        self.codec: int = fourcc
        self.fps: float = fps
        self.frame_size: tuple[int, int] = frame_size
        self.frames: list[np.ndarray] = []
        self.released: bool = False
        FakeLiveVideoWriter.instances.append(self)

    @staticmethod
    def fourcc(*codes: str) -> int:
        assert codes == ("a", "v", "c", "1")
        return 1234

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class FakeLiveVideoCapture:
    instances: list["FakeLiveVideoCapture"] = []

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.frames: list[np.ndarray] = [
            np.full((4, 6, 3), 1, dtype=np.uint8),
            np.full((4, 6, 3), 2, dtype=np.uint8),
        ]
        self.released: bool = False
        FakeLiveVideoCapture.instances.append(self)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return 12.5
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 6.0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 4.0
        raise AssertionError(f"unexpected capture property {prop_id}")

    def release(self) -> None:
        self.released = True


def _accepted_decision(frame_id: int = 0) -> DetectionDecision:
    return DetectionDecision(
        detection=Detection(
            source_id="live",
            frame_id=frame_id,
            timestamp=10.0 + frame_id,
            class_id=1,
            class_name="fire",
            confidence=0.8 + frame_id / 100.0,
            bbox_xyxy=[0.0, 0.0, 4.0, 4.0],
            bbox_area=16.0,
        ),
        accepted=True,
        reason="accepted",
        timestamp=10.0 + frame_id,
    )


def test_live_status_includes_summary_fields(tmp_path: Path) -> None:
    session = LiveDetectionSession(tmp_path / "results")

    status = session.status()

    assert status["summary_available"] is False
    assert status["summary_url"] == ""
    assert status["completed_reason"] == ""


def test_live_session_accumulates_decisions_and_finalizes_stopped_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})
    decisions_by_frame = [[_accepted_decision(0)], [_accepted_decision(1)]]
    monkeypatch.setattr(session, "_frame_decisions", lambda frame, frame_id, record_timestamp, persistence_timestamp: decisions_by_frame[frame_id])
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(cv2, "imwrite", lambda path, frame: True)

    session.process_browser_frame(_jpeg_bytes())
    session.process_browser_frame(_jpeg_bytes())
    status = session.stop()
    summary = session.result()

    assert status["running"] is False
    assert status["summary_available"] is True
    assert status["summary_url"] == "/live/result"
    assert status["completed_reason"] == "stopped"
    assert summary is not None
    assert summary.completed_reason == "stopped"
    assert summary.frame_count == 2
    assert summary.accepted_count == 2
    assert len(summary.decisions) == 2
    assert summary.accepted_counts_by_class == {"fire": 2}


def test_live_session_stop_is_idempotent_and_preserves_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})
    monkeypatch.setattr(session, "_frame_decisions", lambda frame, frame_id, record_timestamp, persistence_timestamp: [_accepted_decision(frame_id)])
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(cv2, "imwrite", lambda path, frame: True)
    session.process_browser_frame(_jpeg_bytes())

    first_status = session.stop()
    first_summary = session.result()
    second_status = session.stop()

    assert first_status == second_status
    assert session.result() is first_summary
    assert first_summary is not None
    assert first_summary.completed_reason == "stopped"


def test_live_uploaded_video_eof_finalizes_completed_summary_and_releases_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeLiveVideoCapture.instances = []
    FakeLiveVideoWriter.instances = []
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    monkeypatch.setattr(cv2, "VideoCapture", FakeLiveVideoCapture)
    monkeypatch.setattr(cv2, "VideoWriter", FakeLiveVideoWriter)
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(cv2, "imwrite", lambda path, frame: True)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "video_file", "file_path": str(video_path)})
    monkeypatch.setattr(session, "_frame_decisions", lambda frame, frame_id, record_timestamp, persistence_timestamp: [_accepted_decision(frame_id)])

    chunks = list(session.mjpeg_frames())
    summary = session.result()

    assert len(chunks) == 2
    assert session.status_state.running is False
    assert session.capture is None
    assert FakeLiveVideoCapture.instances[0].released is True
    assert FakeLiveVideoWriter.instances[0].released is True
    assert FakeLiveVideoWriter.instances[0].fps == 12.5
    assert FakeLiveVideoWriter.instances[0].frame_size == (6, 4)
    assert len(FakeLiveVideoWriter.instances[0].frames) == 2
    assert summary is not None
    assert summary.completed_reason == "completed"
    assert summary.output_path == tmp_path / "results" / f"{session.session_id}_sample_annotated.mp4"
    assert summary.output_available is False
    assert summary.frame_count == 2
    assert len(summary.triggered_frame_paths) == 2
    assert all(path.name.startswith(f"{session.session_id}_triggered_frame_") for path in summary.triggered_frame_paths)


def test_live_start_clears_previous_summary_and_accumulated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})
    monkeypatch.setattr(session, "_frame_decisions", lambda frame, frame_id, record_timestamp, persistence_timestamp: [_accepted_decision(frame_id)])
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(cv2, "imwrite", lambda path, frame: True)
    session.process_browser_frame(_jpeg_bytes())
    session.stop()

    status = session.start({"source_type": "webcam"})

    assert status["running"] is True
    assert status["summary_available"] is False
    assert session.result() is None
    assert session.status_state.frame_count == 0
    assert session.status_state.accepted_count == 0



def test_live_stop_after_zero_frames_finalizes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})

    status = session.stop()
    summary = session.result()

    assert status["running"] is False
    assert status["summary_available"] is True
    assert status["summary_url"] == "/live/result"
    assert status["completed_reason"] == "stopped"
    assert summary is not None
    assert summary.completed_reason == "stopped"
    assert summary.frame_count == 0
    assert summary.decisions == []


def test_browser_frame_processing_serializes_with_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_processing = threading.Event()
    release_processing = threading.Event()
    stop_returned = threading.Event()
    errors: list[BaseException] = []

    def blocking_process_frame(frame: np.ndarray) -> np.ndarray:
        entered_processing.set()
        assert release_processing.wait(timeout=2.0)
        session.status_state.frame_count += 1
        return frame

    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    session = LiveDetectionSession(tmp_path / "results")
    session.start({"source_type": "webcam"})
    monkeypatch.setattr(session, "_process_frame", blocking_process_frame)

    def process_frame() -> None:
        try:
            session.process_browser_frame(_jpeg_bytes())
        except BaseException as exc:
            errors.append(exc)

    def stop_session() -> None:
        try:
            session.stop()
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop_returned.set()

    frame_thread = threading.Thread(target=process_frame)
    stop_thread = threading.Thread(target=stop_session)
    frame_thread.start()
    assert entered_processing.wait(timeout=2.0)
    stop_thread.start()
    time.sleep(0.05)

    assert stop_returned.is_set() is False

    release_processing.set()
    frame_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)

    assert errors == []
    assert stop_returned.is_set() is True
    summary = session.result()
    assert summary is not None
    assert summary.frame_count == 1
    assert summary.completed_reason == "stopped"


def test_stale_mjpeg_generator_cannot_release_new_session_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SingleFrameCapture(FakeLiveVideoCapture):
        def __init__(self, source: str) -> None:
            super().__init__(source)
            self.frames = [np.full((4, 6, 3), len(FakeLiveVideoCapture.instances), dtype=np.uint8)]

    FakeLiveVideoCapture.instances = []
    FakeLiveVideoWriter.instances = []
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"first")
    second_video.write_bytes(b"second")
    monkeypatch.setattr(live_module, "load_config", _config)
    monkeypatch.setattr(live_module, "YOLOEngine", FakeEngine)
    monkeypatch.setattr(cv2, "VideoCapture", SingleFrameCapture)
    monkeypatch.setattr(cv2, "VideoWriter", FakeLiveVideoWriter)
    monkeypatch.setattr(live_module, "render_detections", lambda frame, detections, accepted: frame)
    monkeypatch.setattr(session := LiveDetectionSession(tmp_path / "results"), "_frame_decisions", lambda frame, frame_id, record_timestamp, persistence_timestamp: [])

    session.start({"source_type": "video_file", "file_path": str(first_video)})
    old_generator = session.mjpeg_frames()
    first_chunk = next(old_generator)
    first_capture = FakeLiveVideoCapture.instances[0]
    session.start({"source_type": "video_file", "file_path": str(second_video)})
    second_capture = FakeLiveVideoCapture.instances[1]

    old_generator.close()

    assert first_chunk.startswith(b"--frame")
    assert first_capture.released is True
    assert session.capture is second_capture
    assert second_capture.released is False
    assert session.status_state.running is True
