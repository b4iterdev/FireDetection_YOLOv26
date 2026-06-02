from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.filtering.behavior_tracker import BehaviorTracker
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.detection_filter import DetectionFilter
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.inputs.image_source import ImageSource
from fire_detection_alarm.inputs.video_source import VideoSource
from fire_detection_alarm.logging.detection_logger import DetectionLogger
from fire_detection_alarm.models.yolo_engine import YOLOEngine


@dataclass
class WebPipelineResult:
    input_path: Path
    output_path: Path
    decisions: list[DetectionDecision]
    accepted_count: int


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

        source = self._make_source(input_path)
        decisions: list[DetectionDecision] = []
        accepted_frames = []
        frame_id = 0

        try:
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
                accepted_detections = [decision.detection for decision in frame_decisions if decision.accepted]
                accepted_frames.append(render_detections(frame, accepted_detections))
                frame_id += 1
        finally:
            source.release()

        output_path = self._write_output(input_path, accepted_frames)
        accepted_count = len([decision for decision in decisions if decision.accepted])
        return WebPipelineResult(input_path, output_path, decisions, accepted_count)

    def _process_frame(
        self,
        frame,
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

    def _make_source(self, input_path: Path):
        if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return ImageSource(str(input_path))
        return VideoSource(str(input_path))

    def _write_output(self, input_path: Path, frames: list[Any]) -> Path:
        suffix = input_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            output_path = self.result_dir / f"{input_path.stem}_annotated{suffix}"
            if frames:
                _ = cv2.imwrite(str(output_path), frames[-1])
            return output_path

        output_path = self.result_dir / f"{input_path.stem}_annotated.mp4"
        if not frames:
            return output_path
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter.fourcc(*"mp4v"), 20.0, (width, height))
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return output_path
