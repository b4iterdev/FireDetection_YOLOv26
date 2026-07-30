import io
import sys
import subprocess
from pathlib import Path
from collections.abc import Iterator

from fire_detection_alarm.web.summary import LiveSummary

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.web.app import LiveSession, create_app


class FakeLiveSession(LiveSession):
    def __init__(self):
        self.started_payload: dict[str, object] | None = None
        self.summary: LiveSummary | None = None
        self.cfg: dict[str, dict[str, object]] = {
            "model": {"image_size": 640},
            "inference": {"max_fps": 5},
        }

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self) -> dict[str, object]:
        return {"running": False}

    def status(self) -> dict[str, object]:
        return {"running": self.started_payload is not None}

    def result(self):
        return getattr(self, "summary", None)

    def mjpeg_frames(self) -> Iterator[bytes]:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"

    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        return encoded_frame


def _decision(
    source_id: str,
    frame_id: int,
    class_name: str,
    confidence: float,
    accepted: bool,
    reason: str,
) -> DetectionDecision:
    class_id = 1 if class_name == "fire" else 0
    return DetectionDecision(
        detection=Detection(
            source_id=source_id,
            frame_id=frame_id,
            timestamp=float(frame_id + 1),
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
            bbox_area=4.0,
        ),
        accepted=accepted,
        reason=reason,
        timestamp=float(frame_id + 1),
    )


def _summary(tmp_path: Path) -> LiveSummary:
    result_path = tmp_path / "results"
    result_path.mkdir(exist_ok=True)
    input_path = tmp_path / "uploads" / "sample.mp4"
    input_path.parent.mkdir(exist_ok=True)
    _ = input_path.write_bytes(b"video")
    output_path = result_path / "sample_annotated.mp4"
    _ = output_path.write_bytes(b"output")
    triggered_path = result_path / "sample_triggered_frame_0.jpg"
    _ = triggered_path.write_bytes(b"frame")
    return LiveSummary(
        source_label="sample.mp4",
        source_type="video_file",
        input_path=input_path,
        output_path=output_path,
        decisions=[
            _decision("sample", 0, "fire", 0.91, True, "accepted"),
            _decision("sample", 1, "smoke", 0.42, False, "low_confidence"),
        ],
        accepted_count=1,
        triggered_frame_paths=[triggered_path],
        frame_count=2,
        processing_seconds=0.25,
        completed_reason="completed",
    )


def test_web_root_redirects_to_live(tmp_path: Path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/live")


def test_process_route_is_removed(tmp_path: Path):
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
    )
    client = app.test_client()

    response = client.post(
        "/process",
        data={"file": (io.BytesIO(b"image"), "sample.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 404


def test_live_result_redirects_to_live_when_summary_unavailable(tmp_path: Path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    response = client.get("/live/result", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/live")


def test_live_result_renders_compatible_summary(tmp_path: Path):
    live_session = FakeLiveSession()
    live_session.summary = _summary(tmp_path)
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    response = client.get("/live/result")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Processing complete" in body
    assert "Accepted alarms" in body
    assert "Frames processed" in body
    assert "Decision audit" in body
    assert "Open annotated output" in body
    assert "Return to live monitor" in body
    assert "Dashboard" not in body
    assert "Firewatch" not in body
    assert "<header" not in body
    assert "Analyze another file" not in body


def test_live_summary_exposes_report_metrics(tmp_path: Path):
    result = _summary(tmp_path)

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
) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _ = (result_dir / "annotated.mp4").write_bytes(b"video-output")
    _ = (result_dir / "triggered.jpg").write_bytes(b"frame-output")
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=result_dir)
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
