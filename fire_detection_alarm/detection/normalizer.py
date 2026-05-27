def normalize_yolo_output(result, source_id, frame_id, timestamp):
    from fire_detection_alarm.detection.schema import Detection

    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        area = (x2 - x1) * (y2 - y1)
        detections.append(
            Detection(
                source_id=source_id,
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=int(cls),
                class_name=names[int(cls)],
                confidence=float(conf),
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                bbox_area=float(area),
            )
        )
    return detections
