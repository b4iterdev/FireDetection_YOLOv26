import io
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from fire_detection_alarm.web.app import create_app
from fire_detection_alarm.web import live as live_module
from fire_detection_alarm.web.live import LiveDetectionSession


class FakeLiveSession:
    def __init__(self):
        self.started_payload: dict[str, object] | None = None
        self.stopped = False
        self.processed_frame: bytes | None = None

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self) -> dict[str, object]:
        self.stopped = True
        return {"running": False}

    def status(self) -> dict[str, object]:
        return {"running": self.started_payload is not None and not self.stopped, "frame_count": 0}

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
        self.model_path = model_path
        self.device = device


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


def _jpeg_bytes() -> bytes:
    frame = np.full((4, 6, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_live_page_renders_controls(tmp_path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/live")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Live Detection Player" in body
    assert "rtsp_url" in body
    assert "Frames analyzed" in body
    assert "Accepted alarms" in body
    assert "Raw session diagnostics" in body
    assert 'aria-live="polite"' in body
    assert "Upload video for live detection" not in body
    assert 'id="live-video-file"' in body
    assert 'type="file"' in body
    assert "navigator.mediaDevices.getUserMedia" in body
    assert "/api/live/upload" in body
    assert "/api/live/frame" in body
    assert 'id="camera_index"' not in body
    assert 'id="file_path"' not in body


def test_live_page_receives_configured_max_fps(tmp_path, monkeypatch):
    captured_context: dict[str, object] = {}

    def fake_render_template(template: str, **context: object) -> str:
        captured_context.update(context)
        return template

    monkeypatch.setattr("fire_detection_alarm.web.app.load_config", _config)
    monkeypatch.setattr("flask.render_template", fake_render_template)
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/live")

    assert response.status_code == 200
    assert captured_context["max_fps"] == 5
    assert captured_context["image_size"] == 640


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


def test_live_stream_route_returns_mjpeg(tmp_path):
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=FakeLiveSession(),
    )
    client = app.test_client()

    response = client.get("/stream.mjpeg")

    assert response.status_code == 200
    assert response.content_type.startswith("multipart/x-mixed-replace")


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


def test_mjpeg_stream_respects_configured_max_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = [frame, frame]

        def read(self) -> tuple[bool, np.ndarray | None]:
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

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
