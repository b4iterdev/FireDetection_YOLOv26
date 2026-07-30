from pathlib import Path
from collections.abc import Iterator
from typing import Protocol, Any
from uuid import uuid4
import importlib

from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.web.live import LiveDetectionSession
from fire_detection_alarm.web.summary import LiveSummary


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class LiveSession(Protocol):
    def start(self, payload: dict[str, object]) -> dict[str, object]:
        ...

    def stop(self) -> dict[str, object]:
        ...

    def status(self) -> dict[str, object]:
        ...

    def result(self) -> LiveSummary | None:
        ...

    def mjpeg_frames(self) -> Iterator[bytes]:
        ...

    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        ...


def create_app(
    upload_dir: str | Path = "outputs/web/uploads",
    result_dir: str | Path = "outputs/web/results",
    live_session: LiveSession | None = None,
) -> Any:
    flask = importlib.import_module("flask")
    werkzeug_utils = importlib.import_module("werkzeug.utils")
    app = flask.Flask(__name__)
    upload_path = Path(upload_dir).resolve()
    result_path = Path(result_dir).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    live = live_session or LiveDetectionSession(result_path)
    cfg = load_config()
    max_fps = int(cfg.get("inference", {}).get("max_fps", 5))
    image_size = int(cfg.get("model", {}).get("image_size", 640))

    @app.get("/")
    def index() -> Any:
        return flask.redirect(flask.url_for("live_page"))

    @app.get("/outputs/<path:filename>")
    def outputs(filename: str) -> Any:
        return flask.send_from_directory(result_path, filename)

    @app.get("/live")
    def live_page() -> Any:
        return flask.render_template(
            "live.html",
            show_source_controls=True,
            max_fps=max_fps,
            image_size=image_size,
        )

    @app.get("/live/result")
    def live_result() -> Any:
        result = live.result()
        if result is None:
            return flask.redirect(flask.url_for("live_page"))
        return flask.render_template("result.html", result=result, output_name=result.output_path.name)

    @app.post("/api/live/start")
    def live_start() -> Any:
        payload = flask.request.get_json(silent=True) or {}
        if payload.get("source_type") not in {"webcam", "rtsp"}:
            return {"error": "source_type must be webcam or rtsp"}, 400
        try:
            return live.start(payload)
        except ValueError as exc:
            return {"error": str(exc)}, 400

    @app.post("/api/live/upload")
    def live_upload() -> Any:
        uploaded_file = flask.request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return {"error": "Choose a video file."}, 400

        original_filename = werkzeug_utils.secure_filename(uploaded_file.filename)
        if original_filename == "" or Path(original_filename).suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return {"error": "Upload a supported video file."}, 400

        filename = f"{uuid4().hex}_{original_filename}"
        input_path = (upload_path / filename).resolve()
        try:
            _ = input_path.relative_to(upload_path)
        except ValueError:
            return {"error": "Upload path is invalid."}, 400

        uploaded_file.save(input_path)
        try:
            return live.start({"source_type": "video_file", "file_path": str(input_path)})
        except ValueError as exc:
            return {"error": str(exc)}, 400

    @app.post("/api/live/frame")
    def live_frame() -> Any:
        try:
            annotated_frame = live.process_browser_frame(flask.request.get_data())
        except ValueError as exc:
            return {"error": str(exc)}, 400
        return flask.Response(annotated_frame, mimetype="image/jpeg")

    @app.post("/api/live/stop")
    def live_stop() -> Any:
        return live.stop()

    @app.get("/api/live/status")
    def live_status() -> Any:
        return live.status()

    @app.get("/stream.mjpeg")
    def stream() -> Any:
        status = live.status()
        if not status.get("running") or status.get("source_type") == "webcam":
            return {"error": "live stream is not available"}, 409
        return flask.Response(
            live.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app
