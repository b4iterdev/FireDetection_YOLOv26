import io
import sys
import subprocess
from pathlib import Path
from collections.abc import Iterator

import pytest

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.web.app import LiveSession, create_app
from fire_detection_alarm.web.pipeline import WebPipelineResult


class FakePipelineRunner:
    def __init__(self):
        self.calls: list[tuple[Path, str]] = []

    def run(self, input_path: Path, model_path: str) -> WebPipelineResult:
        self.calls.append((input_path, model_path))
        output_path = input_path.parent / "annotated.jpg"
        _ = output_path.write_bytes(b"fake-image")
        return WebPipelineResult(
            input_path=input_path,
            output_path=output_path,
            decisions=[
                DetectionDecision(
                    detection=Detection(
                        source_id=input_path.stem,
                        frame_id=0,
                        timestamp=1.0,
                        class_id=1,
                        class_name="fire",
                        confidence=0.91,
                        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                        bbox_area=4.0,
                    ),
                    accepted=True,
                    reason="accepted",
                    timestamp=1.0,
                ),
                DetectionDecision(
                    detection=Detection(
                        source_id=input_path.stem,
                        frame_id=1,
                        timestamp=2.0,
                        class_id=0,
                        class_name="smoke",
                        confidence=0.42,
                        bbox_xyxy=[5.0, 6.0, 7.0, 8.0],
                        bbox_area=4.0,
                    ),
                    accepted=False,
                    reason="low_confidence",
                    timestamp=2.0,
                ),
            ],
            accepted_count=1,
            triggered_frame_paths=[input_path.parent / "triggered.jpg"],
            frame_count=2,
            processing_seconds=0.25,
        )


class FakeLiveSession(LiveSession):
    def __init__(self):
        self.started_payload: dict[str, object] | None = None

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self) -> dict[str, object]:
        return {"running": False}

    def status(self) -> dict[str, object]:
        return {"running": self.started_payload is not None}

    def mjpeg_frames(self) -> Iterator[bytes]:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"

    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        return encoded_frame


def test_web_index_renders_live_upload_player(tmp_path: Path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/")

    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Upload image or video" in body
    assert "Run detection" in body
    assert "models/fire_yolov26.pt" in body
    assert 'aria-live="polite"' in body
    assert "Advanced model settings" in body
    assert "Open live monitor" in body
    assert "Live Detection Player" not in body


def test_web_upload_processes_file_with_pipeline_runner(tmp_path: Path):
    pipeline_runner = FakePipelineRunner()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        pipeline_runner=pipeline_runner,
    )
    client = app.test_client()

    response = client.post(
        "/process",
        data={
            "model_path": "models/fire_yolov26.pt",
            "file": (io.BytesIO(b"image"), "sample.jpg"),
        },
        content_type="multipart/form-data",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Processing complete" in body
    assert "Accepted alarms" in body
    assert "Frames processed" in body
    assert "Rejected candidates" in body
    assert "Decision reasons" in body
    assert "Decision audit" in body
    assert "Open annotated output" in body
    assert pipeline_runner.calls[0][0].suffix == ".jpg"
    assert pipeline_runner.calls[0][1] == "models/fire_yolov26.pt"


def test_video_upload_processes_file_with_pipeline_runner_without_starting_live(tmp_path: Path):
    pipeline_runner = FakePipelineRunner()
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        pipeline_runner=pipeline_runner,
        live_session=live_session,
    )
    client = app.test_client()

    response = client.post(
        "/process",
        data={"file": (io.BytesIO(b"video"), "sample.mp4")},
        content_type="multipart/form-data",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Processing complete" in body
    assert "Live Detection Player" not in body
    assert live_session.started_payload is None
    assert pipeline_runner.calls[0][0].suffix == ".mp4"


def test_web_upload_rejects_missing_file_with_dashboard_error(tmp_path: Path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.post("/process", data={}, content_type="multipart/form-data")
    body = response.get_data(as_text=True)

    assert response.status_code == 400
    assert "Choose an image or video file." in body
    assert "Run detection" in body


def test_web_upload_rejects_unsupported_file_before_saving_or_processing(tmp_path: Path):
    pipeline_runner = FakePipelineRunner()
    upload_dir = tmp_path / "uploads"
    app = create_app(
        upload_dir=upload_dir,
        result_dir=tmp_path / "results",
        pipeline_runner=pipeline_runner,
    )
    client = app.test_client()

    response = client.post(
        "/process",
        data={"file": (io.BytesIO(b"text"), "sample.txt")},
        content_type="multipart/form-data",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 400
    assert "Upload a supported image or video file." in body
    assert "Run detection" in body
    assert pipeline_runner.calls == []
    assert list(upload_dir.iterdir()) == []


def test_web_pipeline_result_exposes_report_metrics(tmp_path: Path):
    result = FakePipelineRunner().run(tmp_path / "sample.mp4", "models/fire_yolov26.pt")

    assert result.input_filename == "sample.mp4"
    assert result.input_type == "video"
    assert result.output_available is True
    assert result.rejected_count == 1
    assert result.triggered_frame_count == 1
    assert result.reason_counts == {"accepted": 1, "low_confidence": 1}
    assert result.accepted_counts_by_class == {"fire": 1}
    assert result.max_confidence == 0.91


def test_outputs_serves_relative_result_directory_from_process_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fire_detection_alarm.web.app.load_config",
        lambda: {"model": {"path": "models/fire_yolov26.pt"}},
    )
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "annotated.mp4").write_bytes(b"video-output")
    (result_dir / "triggered.jpg").write_bytes(b"frame-output")
    app = create_app(upload_dir="uploads", result_dir="results", log_dir="logs")
    client = app.test_client()

    video_response = client.get("/outputs/annotated.mp4")
    frame_response = client.get("/outputs/triggered.jpg")

    assert video_response.status_code == 200
    assert video_response.data == b"video-output"
    assert video_response.content_type == "video/mp4"
    assert frame_response.status_code == 200
    assert frame_response.data == b"frame-output"
    assert frame_response.content_type == "image/jpeg"


def test_web_app_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/web_app.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
