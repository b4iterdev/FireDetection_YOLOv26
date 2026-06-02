from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections.abc import Generator

import cv2

from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.filtering.behavior_tracker import BehaviorTracker
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.detection_filter import DetectionFilter
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.models.yolo_engine import YOLOEngine


@dataclass
class LiveStatus:
    running: bool = False
    source_type: str = ""
    frame_count: int = 0
    accepted_count: int = 0
    latest_reason: str = ""
    error: str = ""


class LiveDetectionSession:
    def __init__(self) -> None:
        self.status_state = LiveStatus()
        self.capture = None
        self.engine = None
        self.detection_filter = None
        self.behavior_tracker = None
        self.temporal_filter = None
        self.cfg = None

    def start(self, payload: dict[str, object]) -> dict[str, object]:
        source = self._source_from_payload(payload)
        self.stop()
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
        self.capture = cv2.VideoCapture(source)
        self.status_state = LiveStatus(running=True, source_type=str(payload["source_type"]))
        return self.status()

    def stop(self) -> dict[str, object]:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.status_state.running = False
        return self.status()

    def status(self) -> dict[str, object]:
        return asdict(self.status_state)

    def mjpeg_frames(self) -> Generator[bytes, None, None]:
        while self.status_state.running and self.capture is not None:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.status_state.running = False
                break

            annotated = self._process_frame(frame)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"

    def _process_frame(self, frame):
        assert self.engine is not None
        assert self.cfg is not None
        assert self.detection_filter is not None
        assert self.behavior_tracker is not None
        assert self.temporal_filter is not None
        frame_id = self.status_state.frame_count
        timestamp = time.time()
        results = self.engine.predict(
            frame,
            conf=self.cfg["inference"]["confidence_threshold"],
            iou=self.cfg["inference"]["iou_threshold"],
            imgsz=self.cfg["model"].get("image_size", 640),
        )
        detections = normalize_yolo_output(results[0], "live", frame_id, timestamp)
        decisions: list[DetectionDecision] = []

        static_decisions = [self.detection_filter.check(detection, frame.shape) for detection in detections]
        decisions.extend(decision for decision in static_decisions if not decision.accepted)
        statically_accepted = [decision.detection for decision in static_decisions if decision.accepted]
        behavior_decisions = self.behavior_tracker.check(statically_accepted, frame.shape)
        decisions.extend(decision for decision in behavior_decisions if not decision.accepted)
        post_filter_detections = [decision.detection for decision in behavior_decisions if decision.accepted]

        temporally_accepted = self.temporal_filter.check("live", bool(post_filter_detections), timestamp)
        for detection in post_filter_detections:
            if not temporally_accepted:
                decisions.append(DetectionDecision(detection, False, "not_persistent", timestamp))
                continue
            decisions.append(DetectionDecision(detection, True, "accepted", timestamp))

        accepted_detections = [decision.detection for decision in decisions if decision.accepted]
        self.status_state.frame_count += 1
        self.status_state.accepted_count += len(accepted_detections)
        if decisions:
            self.status_state.latest_reason = decisions[-1].reason
        return render_detections(frame, accepted_detections)

    def _source_from_payload(self, payload: dict[str, object]):
        source_type = payload.get("source_type")
        if source_type == "webcam":
            return int(str(payload.get("camera_index", 0)))
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
