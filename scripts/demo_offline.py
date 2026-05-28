# pyright: reportUnknownVariableType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportArgumentType=false
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
        detections = normalize_yolo_output(results[0], "demo", frame_id, timestamp)
        static_decisions = [
            detection_filter.check(detection, frame.shape)
            for detection in detections
        ]
        static_accepted = [
            decision.detection
            for decision in static_decisions
            if decision.accepted
        ]
        temporally_accepted = temporal_filter.check(
            "demo",
            bool(static_accepted),
            timestamp=timestamp,
        )

        final_decisions = []
        for decision in static_decisions:
            if not decision.accepted:
                final_decisions.append(decision)
                continue

            if not temporally_accepted:
                final_decisions.append(
                    DetectionDecision(decision.detection, False, "not_persistent", timestamp)
                )
                continue

            if not cooldown.check(decision.detection.source_id, timestamp):
                final_decisions.append(
                    DetectionDecision(decision.detection, False, "cooldown_active", timestamp)
                )
                continue

            final_decisions.append(
                DetectionDecision(decision.detection, True, "accepted", timestamp)
            )

        for decision in final_decisions:
            detection_logger.write(decision)

        accepted_detections = [
            decision.detection
            for decision in final_decisions
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
