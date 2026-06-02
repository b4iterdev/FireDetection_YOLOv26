from fire_detection_alarm.web.app import create_app


class FakeLiveSession:
    def __init__(self):
        self.started_payload = None
        self.stopped = False

    def start(self, payload):
        self.started_payload = payload
        return {"running": True, "source_type": payload["source_type"]}

    def stop(self):
        self.stopped = True
        return {"running": False}

    def status(self):
        return {"running": self.started_payload is not None and not self.stopped, "frame_count": 0}

    def mjpeg_frames(self):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"


def test_live_page_renders_controls(tmp_path):
    app = create_app(upload_dir=tmp_path / "uploads", result_dir=tmp_path / "results")
    client = app.test_client()

    response = client.get("/live")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Live Detection Player" in body
    assert "rtsp_url" in body


def test_live_start_accepts_webcam_video_file_and_rtsp(tmp_path):
    live_session = FakeLiveSession()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
        live_session=live_session,
    )
    client = app.test_client()

    for payload in (
        {"source_type": "webcam", "camera_index": 0},
        {"source_type": "video_file", "file_path": "sample.mp4"},
        {"source_type": "rtsp", "rtsp_url": "rtsp://example.local/stream"},
    ):
        response = client.post("/api/live/start", json=payload)
        assert response.status_code == 200
        assert response.get_json()["running"] is True


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
