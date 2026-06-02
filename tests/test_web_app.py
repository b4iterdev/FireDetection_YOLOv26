import io
import sys
import subprocess
from pathlib import Path

from fire_detection_alarm.web.app import create_app
from fire_detection_alarm.web.pipeline import WebPipelineResult


class FakePipelineRunner:
    def run(self, input_path: Path, model_path: str) -> WebPipelineResult:
        output_path = input_path.parent / "annotated.jpg"
        output_path.write_bytes(b"fake-image")
        return WebPipelineResult(
            input_path=input_path,
            output_path=output_path,
            decisions=[],
            accepted_count=0,
        )


class FakeLiveSession:
    def __init__(self):
        self.started_payload = None

    def start(self, payload):
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self):
        return {"running": False}

    def status(self):
        return {"running": self.started_payload is not None}

    def mjpeg_frames(self):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"


def test_web_index_renders_upload_form(tmp_path: Path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/")

    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Upload image or video" in body
    assert "model_path" in body


def test_web_upload_processes_file_with_pipeline_runner(tmp_path: Path):
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        pipeline_runner=FakePipelineRunner(),
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


def test_video_upload_starts_live_playback_instead_of_batch_processing(tmp_path: Path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        pipeline_runner=FakePipelineRunner(),
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
    assert "Live Detection Player" in body
    assert "Processing complete" not in body
    assert live_session.started_payload["source_type"] == "video_file"


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
