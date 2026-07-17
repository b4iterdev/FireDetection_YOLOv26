from dataclasses import dataclass, field
from pathlib import Path
import time
from collections import Counter
from collections.abc import Iterable

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
from fire_detection_alarm.inputs.image_source import ImageSource
from fire_detection_alarm.inputs.video_source import DEFAULT_VIDEO_FPS, VideoSource
from fire_detection_alarm.logging.detection_logger import DetectionLogger
from fire_detection_alarm.models.yolo_engine import YOLOEngine


@dataclass
class WebPipelineResult:
    input_path: Path
    output_path: Path
    decisions: list[DetectionDecision]
    accepted_count: int
    triggered_frame_paths: list[Path] = field(default_factory=list)
    frame_count: int = 0
    processing_seconds: float = 0.0

    @property
    def rejected_count(self) -> int:
        return len([decision for decision in self.decisions if not decision.accepted])

    @property
    def triggered_frame_count(self) -> int:
        return len(self.triggered_frame_paths)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(decision.reason for decision in self.decisions))

    @property
    def accepted_counts_by_class(self) -> dict[str, int]:
        return dict(
            Counter(
                decision.detection.class_name
                for decision in self.decisions
                if decision.accepted
            )
        )

    @property
    def max_confidence(self) -> float:
        confidences = [decision.detection.confidence for decision in self.decisions]
        if not confidences:
            return 0.0
        return max(confidences)

    @property
    def input_filename(self) -> str:
        return self.input_path.name

    @property
    def input_type(self) -> str:
        suffix = self.input_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            return "image"
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            return "video"
        return "unknown"

    @property
    def output_available(self) -> bool:
        return self.output_path.exists()


class WebPipelineRunner:
    result_dir: Path
    log_dir: Path

    def __init__(self, result_dir: str | Path, log_dir: str | Path):
        self.result_dir = Path(result_dir)
        self.log_dir = Path(log_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_path: Path, model_path: str) -> WebPipelineResult:
        cfg = load_config()
        engine = YOLOEngine(model_path, device=cfg["model"]["device"])
        detection_filter = DetectionFilter(
            allowed_class_ids=cfg["classes"]["allowed"],
            min_confidence=cfg["inference"]["confidence_threshold"],
            min_bbox_area_ratio=cfg["filtering"]["min_bbox_area_ratio"],
        )
        behavior_tracker = BehaviorTracker(
            min_track_frames=cfg["behavior_tracking"]["min_track_frames"],
            max_stable_growth_ratio=cfg["behavior_tracking"]["max_stable_growth_ratio"],
            max_non_hazard_area_ratio=cfg["behavior_tracking"]["max_non_hazard_area_ratio"],
            min_growth_ratio=cfg["behavior_tracking"]["min_growth_ratio"],
        )
        temporal_filter = TemporalFilter(
            min_seconds=cfg["filtering"]["min_persistence_seconds"],
            min_frames=cfg["filtering"]["min_consecutive_frames"],
        )
        logger = DetectionLogger(self.log_dir / f"{input_path.stem}.jsonl")

        processing_started = time.perf_counter()
        decisions: list[DetectionDecision] = []
        image_frame: np.ndarray | None = None
        triggered_frame_paths: list[Path] = []
        output_path = self._output_path(input_path)
        is_image = self._is_image_input(input_path)
        source: ImageSource | VideoSource | None = None
        writer: cv2.VideoWriter | None = None
        frame_id = 0

        try:
            source = self._make_source(input_path)
            video_fps = source.fps() if isinstance(source, VideoSource) else DEFAULT_VIDEO_FPS
            while True:
                ret, frame = source.read()
                if not ret or frame is None:
                    break

                frame_decisions = self._process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    source_id=input_path.stem,
                    engine=engine,
                    cfg=cfg,
                    detection_filter=detection_filter,
                    behavior_tracker=behavior_tracker,
                    temporal_filter=temporal_filter,
                )
                for decision in frame_decisions:
                    logger.write(decision)
                decisions.extend(frame_decisions)
                all_detections = self._unique_detections(decision.detection for decision in frame_decisions)
                accepted_detections = [decision.detection for decision in frame_decisions if decision.accepted]
                annotated_frame = render_detections(frame, all_detections, accepted_detections)
                if is_image:
                    image_frame = annotated_frame
                else:
                    if writer is None:
                        height, width = annotated_frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter.fourcc(*"avc1"),
                            video_fps,
                            (width, height),
                        )
                    writer.write(annotated_frame)
                if accepted_detections:
                    triggered_frame_paths.append(self._write_triggered_frame(input_path, frame_id, annotated_frame))
                frame_id += 1
        finally:
            if writer is not None:
                writer.release()
            if source is not None:
                source.release()

        if is_image and image_frame is not None:
            _ = cv2.imwrite(str(output_path), image_frame)
        processing_seconds = time.perf_counter() - processing_started
        accepted_count = len([decision for decision in decisions if decision.accepted])
        return WebPipelineResult(
            input_path=input_path,
            output_path=output_path,
            decisions=decisions,
            accepted_count=accepted_count,
            triggered_frame_paths=triggered_frame_paths,
            frame_count=frame_id,
            processing_seconds=processing_seconds,
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        source_id: str,
        engine: YOLOEngine,
        cfg,
        detection_filter: DetectionFilter,
        behavior_tracker: BehaviorTracker,
        temporal_filter: TemporalFilter,
    ) -> list[DetectionDecision]:
        results = engine.predict(
            frame,
            conf=cfg["inference"]["confidence_threshold"],
            iou=cfg["inference"]["iou_threshold"],
            imgsz=cfg["model"].get("image_size", 640),
        )
        timestamp = time.time()
        detections = normalize_yolo_output(results[0], source_id, frame_id, timestamp)
        pipeline_decisions: list[DetectionDecision] = []

        static_decisions = [detection_filter.check(detection, frame.shape) for detection in detections]
        pipeline_decisions.extend(decision for decision in static_decisions if not decision.accepted)

        statically_accepted = [decision.detection for decision in static_decisions if decision.accepted]
        behavior_decisions = behavior_tracker.check(statically_accepted, frame.shape)
        pipeline_decisions.extend(decision for decision in behavior_decisions if not decision.accepted)
        post_filter_detections = [decision.detection for decision in behavior_decisions if decision.accepted]

        temporally_accepted = temporal_filter.check(source_id, bool(post_filter_detections), timestamp)
        for detection in post_filter_detections:
            if not temporally_accepted:
                pipeline_decisions.append(DetectionDecision(detection, False, "not_persistent", timestamp))
                continue
            pipeline_decisions.append(DetectionDecision(detection, True, "accepted", timestamp))

        return pipeline_decisions

    def _make_source(self, input_path: Path) -> ImageSource | VideoSource:
        if self._is_image_input(input_path):
            return ImageSource(str(input_path))
        return VideoSource(str(input_path))

    def _is_image_input(self, input_path: Path) -> bool:
        return input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    def _output_path(self, input_path: Path) -> Path:
        suffix = input_path.suffix.lower()
        if self._is_image_input(input_path):
            return self.result_dir / f"{input_path.stem}_annotated{suffix}"
        return self.result_dir / f"{input_path.stem}_annotated.mp4"

    def _write_triggered_frame(self, input_path: Path, frame_id: int, frame: np.ndarray) -> Path:
        output_path = self.result_dir / f"{input_path.stem}_triggered_frame_{frame_id}.jpg"
        _ = cv2.imwrite(str(output_path), frame)
        return output_path

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
