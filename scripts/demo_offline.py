import cv2
import time
import argparse
from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.models.yolo_engine import YOLOEngine
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.inputs.video_source import VideoSource
from fire_detection_alarm.inputs.image_source import ImageSource

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/fire_yolov26.pt")
    args = parser.parse_args()

    cfg = load_config()
    engine = YOLOEngine(args.model, device=cfg["model"]["device"])
    
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        source = ImageSource(args.input)
    else:
        source = VideoSource(args.input)
    
    frame_id = 0
    while True:
        ret, frame = source.read()
        if not ret:
            break
        
        results = engine.predict(
            frame, 
            conf=cfg["inference"]["confidence_threshold"],
            iou=cfg["inference"]["iou_threshold"]
        )
        
        detections = normalize_yolo_output(results[0], "demo", frame_id, time.time())
        frame = render_detections(frame, detections)
        
        cv2.imshow("Fire Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1
    
    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
