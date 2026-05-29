import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import time
import argparse
from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.models.yolo_engine import YOLOEngine
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.inputs.video_source import VideoSource
from fire_detection_alarm.inputs.image_source import ImageSource
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.filtering.detection_filter import DetectionFilter
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter
from fire_detection_alarm.filtering.cooldown import CooldownTracker
from fire_detection_alarm.filtering.falsenet_filter import FalseNetFilter
from fire_detection_alarm.logging.detection_logger import DetectionLogger


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--input", required=True)
    _ = parser.add_argument("--model", default="models/fire_yolov26.pt")
    args = parser.parse_args()

    cfg = load_config()
    engine = YOLOEngine(args.model, device=cfg["model"]["device"])
    detection_filter = DetectionFilter(
        allowed_class_ids=cfg["classes"]["allowed"],
        min_confidence=cfg["inference"]["confidence_threshold"],
        min_bbox_area_ratio=cfg["filtering"]["min_bbox_area_ratio"],
    )
    temporal_filter = TemporalFilter(
        min_seconds=cfg["filtering"]["min_persistence_seconds"],
        min_frames=cfg["filtering"]["min_consecutive_frames"],
    )
    cooldown = CooldownTracker(
        cooldown_seconds=cfg["filtering"]["alarm_cooldown_seconds"]
    )
    
    falsenet_filter = None
    if cfg.get("falsenet", {}).get("enabled", False):
        falsenet_filter = FalseNetFilter(
            threat_threshold=cfg["falsenet"]["threat_threshold"]
        )

    log_path = cfg.get("logging", {}).get("detections_path", "outputs/detections.jsonl")
    detection_logger = DetectionLogger(log_path)
    
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        source = ImageSource(args.input)
    else:
        source = VideoSource(args.input)
    
    frame_id = 0
    while True:
        ret, frame = source.read()
        if not ret or frame is None:
            break
        
        results = engine.predict(
            frame, 
            conf=cfg["inference"]["confidence_threshold"],
            iou=cfg["inference"]["iou_threshold"]
        )
        
        timestamp = time.time()
        yolo_detections = normalize_yolo_output(results[0], "demo", frame_id, timestamp)
        
        pipeline_decisions = []

        static_decisions = [detection_filter.check(d, frame.shape) for d in yolo_detections]
        pipeline_decisions.extend(d for d in static_decisions if not d.accepted)
        
        statically_accepted_detections = [d.detection for d in static_decisions if d.accepted]

        if falsenet_filter:
            falsenet_decisions = falsenet_filter.classify(frame, statically_accepted_detections)
            pipeline_decisions.extend(d for d in falsenet_decisions if not d.accepted)
            
            post_falsenet_detections = [d.detection for d in falsenet_decisions if d.accepted]
        else:
            post_falsenet_detections = statically_accepted_detections

        temporally_accepted = temporal_filter.check(
            "demo",
            bool(post_falsenet_detections),
            timestamp=timestamp,
        )

        for detection in post_falsenet_detections:
            if not temporally_accepted:
                pipeline_decisions.append(
                    DetectionDecision(detection, False, "not_persistent", timestamp)
                )
                continue

            if not cooldown.check(detection.source_id, timestamp):
                pipeline_decisions.append(
                    DetectionDecision(detection, False, "cooldown_active", timestamp)
                )
                continue
            
            pipeline_decisions.append(
                DetectionDecision(detection, True, "accepted", timestamp)
            )

        for decision in pipeline_decisions:
            detection_logger.write(decision)

        accepted_detections = [
            decision.detection
            for decision in pipeline_decisions
            if decision.accepted
        ]
        frame = render_detections(frame, accepted_detections)
        
        cv2.imshow("Fire Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1
    
    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

