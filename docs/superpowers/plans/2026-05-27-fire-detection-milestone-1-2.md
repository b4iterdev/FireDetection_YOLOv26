# Fire Detection Model Demo & Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Milestone 1 (Model Demo) and Milestone 2 (Filtering) to run YOLOv26 fire detection on images/videos with basic false alarm filtering.

**Architecture:** A modular pipeline: Input Source -> Frame Reader -> YOLOv26 Inference -> Normalization -> Persistence/Temporal Filtering -> Output. Each stage is isolated for future expansion (RTSP, FalseNet).

**Tech Stack:** Python 3.x, Ultralytics YOLOv11/YOLOv26 (via ultralytics package), OpenCV, PyYAML, Pytest.

---

### Task 1: Project Skeleton and Configuration

**Files:**
- Create: `fire_detection_alarm/app/config.py`
- Create: `configs/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write config loader test**
```python
import os
import yaml
from fire_detection_alarm.app.config import load_config

def test_load_default_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "default.yaml"
    data = {
        "model": {"path": "models/fire.pt"},
        "inference": {"confidence_threshold": 0.5}
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    
    cfg = load_config(str(config_file))
    assert cfg["model"]["path"] == "models/fire.pt"
    assert cfg["inference"]["confidence_threshold"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py`
Expected: ModuleNotFoundError or ImportError

- [ ] **Step 3: Implement config loader**
```python
import yaml
import os

def load_config(config_path="configs/default.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Create default YAML**
```yaml
model:
  path: "models/fire_yolov26.pt"
  device: "auto"
  image_size: 640

inference:
  confidence_threshold: 0.65
  iou_threshold: 0.45
  max_fps: 5

filtering:
  min_persistence_seconds: 2.0
  min_consecutive_frames: 5
  min_bbox_area_ratio: 0.002
  alarm_cooldown_seconds: 60

classes:
  allowed: [0, 1]  # smoke, fire from D-Fire
```

- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/test_config.py`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add fire_detection_alarm/app/config.py configs/default.yaml tests/test_config.py
git commit -m "feat: add project skeleton and config loader"
```

---

### Task 2: Detection Schema and Normalizer

**Files:**
- Create: `fire_detection_alarm/detection/schema.py`
- Create: `fire_detection_alarm/detection/normalizer.py`
- Test: `tests/test_normalizer.py`

- [ ] **Step 1: Define Detection Schema**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    source_id: str
    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]
    bbox_area: float
```

- [ ] **Step 2: Write Normalizer test**
```python
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.schema import Detection

def test_normalize_yolo_output():
    # Mock Ultralytics result-like object
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [xyxy]
            self.conf = [conf]
            self.cls = [cls]
            self.data = [[*xyxy, conf, cls]]

    class MockResult:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    boxes = MockBox([10, 20, 110, 120], 0.9, 1)
    result = MockResult(boxes, {0: "smoke", 1: "fire"})
    
    detections = normalize_yolo_output(result, "cam1", 100, 1622100000.0)
    assert len(detections) == 1
    d = detections[0]
    assert d.class_name == "fire"
    assert d.bbox_area == 10000.0
```

- [ ] **Step 3: Implement Normalizer**
```python
def normalize_yolo_output(result, source_id, frame_id, timestamp):
    from fire_detection_alarm.detection.schema import Detection
    detections = []
    if result.boxes is None:
        return detections
    
    names = result.names
    for box in result.boxes.data:
        # box.data typically [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = box.tolist()
        area = (x2 - x1) * (y2 - y1)
        detections.append(Detection(
            source_id=source_id,
            frame_id=frame_id,
            timestamp=timestamp,
            class_id=int(cls),
            class_name=names[int(cls)],
            confidence=float(conf),
            bbox_xyxy=[x1, y1, x2, y2],
            bbox_area=float(area)
        ))
    return detections
```

- [ ] **Step 4: Run test and Commit**
Run: `pytest tests/test_normalizer.py`
Expected: PASS
```bash
git add fire_detection_alarm/detection/schema.py fire_detection_alarm/detection/normalizer.py tests/test_normalizer.py
git commit -m "feat: add detection schema and yolo normalizer"
```

---

### Task 3: YOLOv26 Inference Engine Wrapper

**Files:**
- Create: `fire_detection_alarm/models/yolo_engine.py`
- Test: `tests/test_yolo_engine.py`

- [ ] **Step 1: Write Engine test (Mocked YOLO)**
```python
from unittest.mock import MagicMock, patch
from fire_detection_alarm.models.yolo_engine import YOLOEngine

@patch("fire_detection_alarm.models.yolo_engine.YOLO")
def test_engine_predict(mock_yolo):
    engine = YOLOEngine("dummy.pt")
    mock_model = mock_yolo.return_value
    mock_model.predict.return_value = [MagicMock(boxes=MagicMock(data=[]), names={0:"smoke"})]
    
    results = engine.predict("frame.jpg")
    assert len(results) == 1
    mock_model.predict.assert_called_once()
```

- [ ] **Step 2: Implement Engine Wrapper**
```python
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
```

- [ ] **Step 3: Run test and Commit**
Run: `pytest tests/test_yolo_engine.py`
Expected: PASS
```bash
git add fire_detection_alarm/models/yolo_engine.py tests/test_yolo_engine.py
git commit -m "feat: add YOLOv26 inference engine wrapper"
```

---

### Task 4: Image/Video Sources

**Files:**
- Create: `fire_detection_alarm/inputs/base.py`
- Create: `fire_detection_alarm/inputs/image_source.py`
- Create: `fire_detection_alarm/inputs/video_source.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Implement Base and Image Source**
```python
# base.py
from abc import ABC, abstractmethod
class BaseSource(ABC):
    @abstractmethod
    def read(self): pass
    @abstractmethod
    def release(self): pass

# image_source.py
import cv2
class ImageSource(BaseSource):
    def __init__(self, path):
        self.path = path
        self.frame = None
    def read(self):
        if self.frame is not None: return False, None
        self.frame = cv2.imread(self.path)
        return (True, self.frame) if self.frame is not None else (False, None)
    def release(self): pass
```

- [ ] **Step 2: Implement Video Source**
```python
class VideoSource(BaseSource):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
    def read(self):
        return self.cap.read()
    def release(self):
        self.cap.release()
```

- [ ] **Step 3: Commit**
```bash
git add fire_detection_alarm/inputs/*.py tests/test_sources.py
git commit -m "feat: add image and video input sources"
```

---

### Task 5: Basic Renderer

**Files:**
- Create: `fire_detection_alarm/detection/renderer.py`

- [ ] **Step 1: Implement Render Function**
```python
import cv2
def render_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame
```

- [ ] **Step 2: Commit**
```bash
git add fire_detection_alarm/detection/renderer.py
git commit -m "feat: add basic detection renderer"
```

---

### Task 6: False Alarm Filter (Temporal Persistence)

**Files:**
- Create: `fire_detection_alarm/filtering/temporal_filter.py`
- Test: `tests/test_filtering.py`

- [ ] **Step 1: Write Temporal Filter test**
```python
import time
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter

def test_temporal_persistence():
    f = TemporalFilter(min_seconds=1.0)
    # First detection
    assert f.check("cam1", True, timestamp=100.0) is False
    # Half second later
    assert f.check("cam1", True, timestamp=100.5) is False
    # Over 1 second later
    assert f.check("cam1", True, timestamp=101.1) is True
    # If detection lost
    assert f.check("cam1", False, timestamp=101.2) is False
```

- [ ] **Step 2: Implement Temporal Filter**
```python
class TemporalFilter:
    def __init__(self, min_seconds=2.0):
        self.min_seconds = min_seconds
        self.active_starts = {} # source_id -> start_timestamp

    def check(self, source_id, is_detected, timestamp):
        if not is_detected:
            if source_id in self.active_starts:
                del self.active_starts[source_id]
            return False
        
        if source_id not in self.active_starts:
            self.active_starts[source_id] = timestamp
            return False
        
        duration = timestamp - self.active_starts[source_id]
        return duration >= self.min_seconds
```

- [ ] **Step 3: Run test and Commit**
Run: `pytest tests/test_filtering.py`
Expected: PASS
```bash
git add fire_detection_alarm/filtering/temporal_filter.py tests/test_filtering.py
git commit -m "feat: add temporal persistence filter"
```

---

### Task 7: Milestone 1 Demo Script

**Files:**
- Create: `scripts/demo_offline.py`

- [ ] **Step 1: Implement Demo Script**
```python
import cv2
import time
import argparse
from fire_detection_alarm.app.config import load_config
from fire_detection_alarm.models.yolo_engine import YOLOEngine
from fire_detection_alarm.detection.normalizer import normalize_yolo_output
from fire_detection_alarm.detection.renderer import render_detections
from fire_detection_alarm.inputs.video_source import VideoSource

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/fire_yolov26.pt")
    args = parser.parse_args()

    cfg = load_config()
    engine = YOLOEngine(args.model)
    source = VideoSource(args.input)
    
    frame_id = 0
    while True:
        ret, frame = source.read()
        if not ret: break
        
        results = engine.predict(frame, conf=cfg["inference"]["confidence_threshold"])
        detections = normalize_yolo_output(results[0], "demo", frame_id, time.time())
        
        frame = render_detections(frame, detections)
        cv2.imshow("Fire Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_id += 1
    
    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**
```bash
git add scripts/demo_offline.py
git commit -m "feat: add offline demo script for Milestone 1"
```
