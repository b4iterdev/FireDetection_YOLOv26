from pathlib import Path
from typing import Protocol, Any
from uuid import uuid4
import importlib

from fire_detection_alarm.web.pipeline import WebPipelineResult
from fire_detection_alarm.web.pipeline import WebPipelineRunner
from fire_detection_alarm.web.live import LiveDetectionSession


class PipelineRunner(Protocol):
    def run(self, input_path: Path, model_path: str) -> WebPipelineResult:
        ...


class LiveSession(Protocol):
    def start(self, payload: dict[str, object]) -> dict[str, object]:
        ...

    def stop(self) -> dict[str, object]:
        ...

    def status(self) -> dict[str, object]:
        ...

    def mjpeg_frames(self):
        ...


def create_app(
    upload_dir: str | Path = "outputs/web/uploads",
    result_dir: str | Path = "outputs/web/results",
    log_dir: str | Path = "outputs/web/logs",
    pipeline_runner: PipelineRunner | None = None,
    live_session: LiveSession | None = None,
) -> Any:
    flask = importlib.import_module("flask")
    werkzeug_utils = importlib.import_module("werkzeug.utils")
    app = flask.Flask(__name__)
    upload_path = Path(upload_dir)
    result_path = Path(result_dir)
    log_path = Path(log_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    runner = pipeline_runner or WebPipelineRunner(result_path, log_path)
    live = live_session or LiveDetectionSession()

    @app.get("/")
    def index() -> Any:
        return flask.render_template("live.html", show_source_controls=True)

    @app.post("/process")
    def process_upload() -> Any:
        uploaded_file = flask.request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return flask.render_template("index.html", error="Choose an image or video file.", default_model="models/fire_yolov26.pt"), 400

        filename = f"{uuid4().hex}_{werkzeug_utils.secure_filename(uploaded_file.filename)}"
        input_path = upload_path / filename
        uploaded_file.save(input_path)

        if input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            live.start({"source_type": "video_file", "file_path": str(input_path)})
            return flask.render_template("live.html", show_source_controls=False)

        model_path = flask.request.form.get("model_path") or "models/fire_yolov26.pt"
        result = runner.run(input_path, model_path)
        return flask.render_template("result.html", result=result, output_name=result.output_path.name)

    @app.get("/outputs/<path:filename>")
    def outputs(filename: str) -> Any:
        return flask.send_from_directory(result_path, filename)

    @app.get("/live")
    def live_page() -> Any:
        return flask.render_template("live.html", show_source_controls=True)

    @app.post("/api/live/start")
    def live_start() -> Any:
        payload = flask.request.get_json(silent=True) or {}
        if payload.get("source_type") not in {"webcam", "video_file", "rtsp"}:
            return {"error": "source_type must be webcam, video_file, or rtsp"}, 400
        try:
            return live.start(payload)
        except ValueError as exc:
            return {"error": str(exc)}, 400

    @app.post("/api/live/stop")
    def live_stop() -> Any:
        return live.stop()

    @app.get("/api/live/status")
    def live_status() -> Any:
        return live.status()

    @app.get("/stream.mjpeg")
    def stream() -> Any:
        return flask.Response(
            live.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app
