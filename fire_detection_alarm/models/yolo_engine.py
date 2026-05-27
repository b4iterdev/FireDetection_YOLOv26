from ultralytics import YOLO

class YOLOEngine:
    def __init__(self, model_path, device="auto"):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame, conf=0.25, iou=0.45, imgsz=640):
        return self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
