from dataclasses import dataclass, asdict
from pathlib import Path
from uuid import uuid4
import math
import threading
import time
from collections.abc import Generator, Iterable

import cv2
import numpy as np

from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.behavior_tracker import BehaviorTracker
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.detection_filter import DetectionFilter
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.inputs.video_source import DEFAULT_VIDEO_FPS
from fire_detection_alarm.models.yolo_engine import YOLOEngine
from fire_detection_alarm.web.summary import LiveSummary


@dataclass
class LiveStatus:
    running: bool = False
    source_type: str = ""
    frame_count: int = 0
    accepted_count: int = 0
    latest_reason: str = ""
    latest_triggered_frame: str = ""
    error: str = ""
    summary_available: bool = False
    summary_url: str = ""
    completed_reason: str = ""


class LiveDetectionSession:
    def __init__(self, result_dir: str | Path = "outputs/web/results") -> None:
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.status_state = LiveStatus()
        self.capture = None
        self.engine = None
        self.detection_filter = None
        self.behavior_tracker = None
        self.temporal_filter = None
        self.cfg = None
        self.video_file_fps: float | None = None
        self.video_writer = None
        self.video_output_path: Path | None = None
        self.input_path: Path | None = None
        self.source_label = ""
        self.session_id = uuid4().hex
        self.processing_started: float | None = None
        self.session_started = False
        self.decisions: list[DetectionDecision] = []
        self.triggered_frame_paths: list[Path] = []
        self.summary: LiveSummary | None = None
        self.lock = threading.RLock()
        self.generation = 0

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        source_type = str(payload.get("source_type", ""))
        source = self._source_from_payload(payload)
        with self.lock:
            self._release_resources()
            self.generation += 1
            self._initialize_detector()
            self._reset_started_state(source_type, source)
            if source_type != "webcam":
                assert source is not None
                self.capture = cv2.VideoCapture(source)
                if source_type == "video_file":
                    assert self.input_path is not None
                    self.video_file_fps = self._capture_fps()
                    self.video_output_path = self.result_dir / f"{self.session_id}_{self.input_path.stem}_annotated.mp4"
            self.status_state = LiveStatus(running=True, source_type=source_type)
            return self._status_locked()

    def stop(self) -> dict[str, object]:
        with self.lock:
            if self.summary is None and self.session_started:
                self._finalize("stopped")
            self._release_resources()
            self.status_state.running = False
            self._sync_summary_status()
            return self._status_locked()

    def status(self) -> dict[str, object]:
        with self.lock:
            return self._status_locked()

    def result(self) -> LiveSummary | None:
        with self.lock:
            return self.summary

    def mjpeg_frames(self) -> Generator[bytes, None, None]:
        with self.lock:
            generation = self.generation
            assert self.cfg is not None
            max_fps = max(float(self.cfg["inference"].get("max_fps", 5)), 1.0)
        frame_interval = 1.0 / max_fps
        last_frame_started: float | None = None
        try:
            while True:
                with self.lock:
                    if generation != self.generation or not self.status_state.running or self.capture is None:
                        break
                    ret, frame = self.capture.read()
                    if not ret or frame is None:
                        if generation == self.generation and self.status_state.source_type == "video_file":
                            self._finalize("completed")
                        if generation == self.generation:
                            self._release_resources()
                            self.status_state.running = False
                            self._sync_summary_status()
                        break

                now = time.monotonic()
                if last_frame_started is not None:
                    delay = frame_interval - (now - last_frame_started)
                    if delay > 0:
                        time.sleep(delay)
                last_frame_started = time.monotonic()

                with self.lock:
                    if generation != self.generation or not self.status_state.running:
                        break
                    annotated = self._process_frame(frame)
                    ok, encoded = cv2.imencode(".jpg", annotated)
                if not ok:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        finally:
            with self.lock:
                if (
                    generation == self.generation
                    and self.status_state.source_type == "video_file"
                    and self.status_state.running
                ):
                    self._release_resources()
                    self.status_state.running = False
                    self._sync_summary_status()

    def process_browser_frame(self, encoded_frame: bytes) -> bytes:
        with self.lock:
            if not self.status_state.running or self.status_state.source_type != "webcam":
                raise ValueError("webcam session is not running")
            if not encoded_frame:
                raise ValueError("invalid JPEG frame")

            frame_buffer = np.frombuffer(encoded_frame, dtype=np.uint8)
            frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("invalid JPEG frame")

            annotated = self._process_frame(frame)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                raise ValueError("could not encode JPEG frame")
            return encoded.tobytes()

    def _initialize_detector(self) -> None:
        self.cfg = load_config()
        self.engine = YOLOEngine(self.cfg["model"]["path"], device=self.cfg["model"]["device"])
        self.detection_filter = DetectionFilter(
            self.cfg["classes"]["allowed"],
            self.cfg["inference"]["confidence_threshold"],
            self.cfg["filtering"]["min_bbox_area_ratio"],
        )
        self.behavior_tracker = BehaviorTracker(
            self.cfg["behavior_tracking"]["min_track_frames"],
            self.cfg["behavior_tracking"]["max_stable_growth_ratio"],
            self.cfg["behavior_tracking"]["max_non_hazard_area_ratio"],
            self.cfg["behavior_tracking"]["min_growth_ratio"],
        )
        self.temporal_filter = TemporalFilter(
            self.cfg["filtering"]["min_persistence_seconds"],
            self.cfg["filtering"]["min_consecutive_frames"],
        )

    def _reset_started_state(self, source_type: str, source: str | None) -> None:
        self.session_id = uuid4().hex
        self.decisions = []
        self.triggered_frame_paths = []
        self.summary = None
        self.video_writer = None
        self.video_output_path = None
        self.video_file_fps = None
        self.processing_started = time.perf_counter()
        self.session_started = True
        self.input_path = Path(source) if source_type == "video_file" and source is not None else None
        if self.input_path is not None:
            self.source_label = self.input_path.name
        elif source_type == "rtsp" and source is not None:
            self.source_label = source
        else:
            self.source_label = "Browser webcam"

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_id = self.status_state.frame_count
        record_timestamp = time.time()
        persistence_timestamp = self._persistence_timestamp(frame_id, record_timestamp)
        decisions = self._frame_decisions(frame, frame_id, record_timestamp, persistence_timestamp)
        all_detections = self._unique_detections(decision.detection for decision in decisions)
        accepted_detections = [decision.detection for decision in decisions if decision.accepted]
        annotated = render_detections(frame, all_detections, accepted_detections)

        self.decisions.extend(decisions)
        self.status_state.frame_count += 1
        self.status_state.accepted_count += len(accepted_detections)
        if decisions:
            self.status_state.latest_reason = decisions[-1].reason
        if self.status_state.source_type == "video_file":
            self._write_video_output(annotated)
        if accepted_detections:
            triggered_frame_path = self._write_triggered_frame(frame_id, annotated)
            if triggered_frame_path is not None:
                self.triggered_frame_paths.append(triggered_frame_path)
                self.status_state.latest_triggered_frame = f"/outputs/{triggered_frame_path.name}"
        return annotated

    def _frame_decisions(
        self,
        frame: np.ndarray,
        frame_id: int,
        record_timestamp: float,
        persistence_timestamp: float,
    ) -> list[DetectionDecision]:
        assert self.engine is not None
        assert self.cfg is not None
        assert self.detection_filter is not None
        assert self.behavior_tracker is not None
        assert self.temporal_filter is not None
        results = self.engine.predict(
            frame,
            conf=self.cfg["inference"]["confidence_threshold"],
            iou=self.cfg["inference"]["iou_threshold"],
            imgsz=self.cfg["model"].get("image_size", 640),
        )
        detections = normalize_yolo_output(results[0], "live", frame_id, record_timestamp)
        decisions: list[DetectionDecision] = []

        static_decisions = [self.detection_filter.check(detection, frame.shape) for detection in detections]
        decisions.extend(decision for decision in static_decisions if not decision.accepted)
        statically_accepted = [decision.detection for decision in static_decisions if decision.accepted]
        behavior_decisions = self.behavior_tracker.check(statically_accepted, frame.shape)
        decisions.extend(decision for decision in behavior_decisions if not decision.accepted)
        post_filter_detections = [decision.detection for decision in behavior_decisions if decision.accepted]

        temporally_accepted = self.temporal_filter.check("live", bool(post_filter_detections), persistence_timestamp)
        for detection in post_filter_detections:
            if not temporally_accepted:
                decisions.append(DetectionDecision(detection, False, "not_persistent", record_timestamp))
                continue
            decisions.append(DetectionDecision(detection, True, "accepted", record_timestamp))
        return decisions

    def _write_video_output(self, annotated: np.ndarray) -> None:
        assert self.video_output_path is not None
        if self.video_writer is None:
            assert self.video_file_fps is not None
            frame_size = self._capture_frame_size(annotated)
            self.video_writer = cv2.VideoWriter(
                str(self.video_output_path),
                cv2.VideoWriter.fourcc(*"avc1"),
                self.video_file_fps,
                frame_size,
            )
        self.video_writer.write(annotated)

    def _capture_frame_size(self, frame: np.ndarray) -> tuple[int, int]:
        assert self.capture is not None
        width = float(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0:
            return (int(width), int(height))
        frame_height, frame_width = frame.shape[:2]
        return (frame_width, frame_height)

    def _write_triggered_frame(self, frame_id: int, frame: np.ndarray) -> Path | None:
        triggered_frame_path = self.result_dir / f"{self.session_id}_triggered_frame_{frame_id}.jpg"
        if cv2.imwrite(str(triggered_frame_path), frame):
            return triggered_frame_path
        return None

    def _finalize(self, completed_reason: str) -> None:
        if self.summary is not None:
            return
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        processing_started = self.processing_started if self.processing_started is not None else time.perf_counter()
        output_path = self.video_output_path or self.result_dir / f"{self.session_id}_live_output.mp4"
        self.summary = LiveSummary(
            source_label=self.source_label,
            source_type=self.status_state.source_type,
            input_path=self.input_path,
            output_path=output_path,
            decisions=list(self.decisions),
            accepted_count=len([decision for decision in self.decisions if decision.accepted]),
            triggered_frame_paths=list(self.triggered_frame_paths),
            frame_count=self.status_state.frame_count,
            processing_seconds=time.perf_counter() - processing_started,
            completed_reason=completed_reason,
        )
        self._sync_summary_status()

    def _release_resources(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.video_file_fps = None

    def _sync_summary_status(self) -> None:
        self.status_state.summary_available = self.summary is not None
        self.status_state.summary_url = "/live/result" if self.summary is not None else ""
        self.status_state.completed_reason = self.summary.completed_reason if self.summary is not None else ""

    def _status_locked(self) -> dict[str, object]:
        self._sync_summary_status()
        return asdict(self.status_state)

    def _capture_fps(self) -> float:
        assert self.capture is not None
        capture_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if math.isfinite(capture_fps) and capture_fps > 0:
            return capture_fps
        return DEFAULT_VIDEO_FPS

    def _persistence_timestamp(self, frame_id: int, record_timestamp: float) -> float:
        if self.status_state.source_type == "video_file":
            assert self.video_file_fps is not None
            return float(frame_id) / self.video_file_fps
        return record_timestamp

    def _unique_detections(self, detections: Iterable[Detection]) -> list[Detection]:
        unique_detections: list[Detection] = []
        seen_ids: set[int] = set()
        for detection in detections:
            detection_id = id(detection)
            if detection_id in seen_ids:
                continue
            seen_ids.add(detection_id)
            unique_detections.append(detection)
        return unique_detections

    def _source_from_payload(self, payload: dict[str, object]):
        source_type = payload.get("source_type")
        if source_type == "webcam":
            return None
        if source_type == "video_file":
            file_path = Path(str(payload.get("file_path", "")))
            if not file_path.exists():
                raise ValueError("video file does not exist")
            return str(file_path)
        if source_type == "rtsp":
            rtsp_url = str(payload.get("rtsp_url", ""))
            if not rtsp_url.startswith("rtsp://"):
                raise ValueError("rtsp_url must start with rtsp://")
            return rtsp_url
        raise ValueError("source_type must be webcam, video_file, or rtsp")
