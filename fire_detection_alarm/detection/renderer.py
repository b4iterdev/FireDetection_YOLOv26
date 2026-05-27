import cv2

def render_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame
