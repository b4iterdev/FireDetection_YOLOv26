# Milestone 2 Filtering Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Milestone 2 by adding explainable static filtering, temporal frame persistence, cooldown, JSONL decision logging, and offline demo integration.

**Architecture:** Normalized `Detection` objects flow through stateless static filtering, stateful temporal persistence, stateful cooldown, then JSONL logging. The demo script remains a coordinator and renders only accepted detections.

**Tech Stack:** Python, dataclasses, pathlib, json, OpenCV, PyYAML, pytest.

---

## File Structure

- Create `fire_detection_alarm/filtering/decision.py`
  - Owns `DetectionDecision`, the stable accepted/rejected decision schema.
- Create `fire_detection_alarm/filtering/detection_filter.py`
  - Owns stateless class, confidence, and bbox area ratio filtering.
- Modify `fire_detection_alarm/filtering/temporal_filter.py`
  - Adds consecutive-frame persistence while preserving elapsed-second behavior.
- Create `fire_detection_alarm/filtering/cooldown.py`
  - Owns per-source cooldown state.
- Create `fire_detection_alarm/logging/__init__.py`
  - Makes logging package importable.
- Create `fire_detection_alarm/logging/detection_logger.py`
  - Owns JSONL writing for decisions.
- Modify `configs/default.yaml`
  - Adds optional log path config.
- Modify `scripts/demo_offline.py`
  - Wires filtering and logging into the existing offline pipeline.
- Create/modify tests:
  - `tests/test_detection_filter.py`
  - `tests/test_filtering.py`
  - `tests/test_cooldown.py`
  - `tests/test_detection_logger.py`

---

### Task 1: Filtering Decision Schema and Static Detection Filter

**Files:**
- Create: `fire_detection_alarm/filtering/decision.py`
- Create: `fire_detection_alarm/filtering/detection_filter.py`
- Test: `tests/test_detection_filter.py`

- [ ] **Step 1: Write failing tests for static filtering decisions**

Create `tests/test_detection_filter.py`:

```python
from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.detection_filter import DetectionFilter


def make_detection(class_id=1, confidence=0.9, bbox_area=10000.0):
    return Detection(
        source_id="cam1",
        frame_id=1,
        timestamp=100.0,
        class_id=class_id,
        class_name="fire",
        confidence=confidence,
        bbox_xyxy=[10.0, 10.0, 110.0, 110.0],
        bbox_area=bbox_area,
    )


def test_detection_filter_accepts_valid_detection():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )

    decision = detection_filter.check(make_detection(), frame_shape=(1000, 1000, 3))

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.detection.class_id == 1


def test_detection_filter_rejects_disallowed_class():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )

    decision = detection_filter.check(make_detection(class_id=2), frame_shape=(1000, 1000, 3))

    assert decision.accepted is False
    assert decision.reason == "class_not_allowed"


def test_detection_filter_rejects_low_confidence():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )

    decision = detection_filter.check(make_detection(confidence=0.5), frame_shape=(1000, 1000, 3))

    assert decision.accepted is False
    assert decision.reason == "low_confidence"


def test_detection_filter_rejects_small_bbox_ratio():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )

    decision = detection_filter.check(make_detection(bbox_area=100.0), frame_shape=(1000, 1000, 3))

    assert decision.accepted is False
    assert decision.reason == "bbox_too_small"


def test_detection_filter_rejects_missing_frame_shape():
    detection_filter = DetectionFilter(
        allowed_class_ids=[1],
        min_confidence=0.65,
        min_bbox_area_ratio=0.002,
    )

    decision = detection_filter.check(make_detection(), frame_shape=None)

    assert decision.accepted is False
    assert decision.reason == "missing_frame_size"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_detection_filter.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `fire_detection_alarm.filtering.detection_filter`.

- [ ] **Step 3: Implement decision schema and detection filter**

Create `fire_detection_alarm/filtering/decision.py`:

```python
from dataclasses import dataclass
from fire_detection_alarm.detection.schema import Detection


@dataclass
class DetectionDecision:
    detection: Detection
    accepted: bool
    reason: str
    timestamp: float
```

Create `fire_detection_alarm/filtering/detection_filter.py`:

```python
from fire_detection_alarm.filtering.decision import DetectionDecision


class DetectionFilter:
    def __init__(self, allowed_class_ids, min_confidence, min_bbox_area_ratio):
        self.allowed_class_ids = set(allowed_class_ids)
        self.min_confidence = float(min_confidence)
        self.min_bbox_area_ratio = float(min_bbox_area_ratio)

    def check(self, detection, frame_shape):
        if detection.class_id not in self.allowed_class_ids:
            return DetectionDecision(detection, False, "class_not_allowed", detection.timestamp)

        if detection.confidence < self.min_confidence:
            return DetectionDecision(detection, False, "low_confidence", detection.timestamp)

        if frame_shape is None or len(frame_shape) < 2:
            return DetectionDecision(detection, False, "missing_frame_size", detection.timestamp)

        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        if frame_area <= 0:
            return DetectionDecision(detection, False, "missing_frame_size", detection.timestamp)

        bbox_area_ratio = detection.bbox_area / frame_area
        if bbox_area_ratio < self.min_bbox_area_ratio:
            return DetectionDecision(detection, False, "bbox_too_small", detection.timestamp)

        return DetectionDecision(detection, True, "accepted", detection.timestamp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_detection_filter.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
GIT_MASTER=1 git add fire_detection_alarm/filtering/decision.py fire_detection_alarm/filtering/detection_filter.py tests/test_detection_filter.py
GIT_MASTER=1 git commit -m "feat: add static detection filtering"
```

---

### Task 2: Temporal Filter Frame Persistence

**Files:**
- Modify: `fire_detection_alarm/filtering/temporal_filter.py`
- Modify: `tests/test_filtering.py`

- [ ] **Step 1: Replace temporal filter tests with seconds and frame-count coverage**

Replace `tests/test_filtering.py` with:

```python
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter


def test_temporal_persistence_requires_seconds_and_frames():
    temporal_filter = TemporalFilter(min_seconds=1.0, min_frames=3)

    assert temporal_filter.check("cam1", True, timestamp=100.0) is False
    assert temporal_filter.check("cam1", True, timestamp=100.5) is False
    assert temporal_filter.check("cam1", True, timestamp=101.1) is True


def test_temporal_persistence_resets_when_detection_lost():
    temporal_filter = TemporalFilter(min_seconds=1.0, min_frames=2)

    assert temporal_filter.check("cam1", True, timestamp=100.0) is False
    assert temporal_filter.check("cam1", False, timestamp=100.5) is False
    assert temporal_filter.check("cam1", True, timestamp=101.6) is False
    assert temporal_filter.check("cam1", True, timestamp=102.7) is True


def test_temporal_persistence_supports_seconds_only():
    temporal_filter = TemporalFilter(min_seconds=1.0, min_frames=0)

    assert temporal_filter.check("cam1", True, timestamp=100.0) is False
    assert temporal_filter.check("cam1", True, timestamp=101.1) is True


def test_temporal_persistence_supports_frames_only():
    temporal_filter = TemporalFilter(min_seconds=0.0, min_frames=2)

    assert temporal_filter.check("cam1", True, timestamp=100.0) is False
    assert temporal_filter.check("cam1", True, timestamp=100.1) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_filtering.py -q
```

Expected: FAIL with `TypeError` because `TemporalFilter` does not accept `min_frames`.

- [ ] **Step 3: Implement frame-count temporal persistence**

Replace `fire_detection_alarm/filtering/temporal_filter.py` with:

```python
class TemporalFilter:
    def __init__(self, min_seconds=2.0, min_frames=0):
        self.min_seconds = float(min_seconds)
        self.min_frames = int(min_frames)
        self.active_starts = {}
        self.active_counts = {}

    def check(self, source_id, is_detected, timestamp):
        if not is_detected:
            self.active_starts.pop(source_id, None)
            self.active_counts.pop(source_id, None)
            return False

        if source_id not in self.active_starts:
            self.active_starts[source_id] = timestamp
            self.active_counts[source_id] = 1
        else:
            self.active_counts[source_id] += 1

        duration = timestamp - self.active_starts[source_id]
        seconds_passed = self.min_seconds <= 0 or duration >= self.min_seconds
        frames_passed = self.min_frames <= 0 or self.active_counts[source_id] >= self.min_frames
        return seconds_passed and frames_passed
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_filtering.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
GIT_MASTER=1 git add fire_detection_alarm/filtering/temporal_filter.py tests/test_filtering.py
GIT_MASTER=1 git commit -m "feat: add frame persistence filtering"
```

---

### Task 3: Cooldown Tracker

**Files:**
- Create: `fire_detection_alarm/filtering/cooldown.py`
- Test: `tests/test_cooldown.py`

- [ ] **Step 1: Write failing cooldown tests**

Create `tests/test_cooldown.py`:

```python
from fire_detection_alarm.filtering.cooldown import CooldownTracker


def test_cooldown_accepts_first_detection():
    cooldown = CooldownTracker(cooldown_seconds=60)

    assert cooldown.check("cam1", timestamp=100.0) is True


def test_cooldown_rejects_detection_inside_window():
    cooldown = CooldownTracker(cooldown_seconds=60)

    cooldown.check("cam1", timestamp=100.0)

    assert cooldown.check("cam1", timestamp=120.0) is False


def test_cooldown_accepts_detection_after_window():
    cooldown = CooldownTracker(cooldown_seconds=60)

    cooldown.check("cam1", timestamp=100.0)
    assert cooldown.check("cam1", timestamp=161.0) is True


def test_cooldown_tracks_sources_independently():
    cooldown = CooldownTracker(cooldown_seconds=60)

    cooldown.check("cam1", timestamp=100.0)
    assert cooldown.check("cam2", timestamp=120.0) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cooldown.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `fire_detection_alarm.filtering.cooldown`.

- [ ] **Step 3: Implement cooldown tracker**

Create `fire_detection_alarm/filtering/cooldown.py`:

```python
class CooldownTracker:
    def __init__(self, cooldown_seconds=60):
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_accepted_at = {}

    def check(self, source_id, timestamp):
        last_accepted = self.last_accepted_at.get(source_id)
        if last_accepted is None:
            self.last_accepted_at[source_id] = timestamp
            return True

        if timestamp - last_accepted >= self.cooldown_seconds:
            self.last_accepted_at[source_id] = timestamp
            return True

        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cooldown.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
GIT_MASTER=1 git add fire_detection_alarm/filtering/cooldown.py tests/test_cooldown.py
GIT_MASTER=1 git commit -m "feat: add detection cooldown tracking"
```

---

### Task 4: JSONL Detection Decision Logger

**Files:**
- Create: `fire_detection_alarm/logging/__init__.py`
- Create: `fire_detection_alarm/logging/detection_logger.py`
- Test: `tests/test_detection_logger.py`

- [ ] **Step 1: Write failing logger test**

Create `tests/test_detection_logger.py`:

```python
import json
from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision
from fire_detection_alarm.logging.detection_logger import DetectionLogger


def test_detection_logger_writes_jsonl_record(tmp_path):
    log_path = tmp_path / "detections.jsonl"
    logger = DetectionLogger(str(log_path))
    detection = Detection(
        source_id="cam1",
        frame_id=7,
        timestamp=123.4,
        class_id=1,
        class_name="fire",
        confidence=0.91,
        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        bbox_area=4.0,
    )
    decision = DetectionDecision(
        detection=detection,
        accepted=True,
        reason="accepted",
        timestamp=123.5,
    )

    logger.write(decision)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records == [
        {
            "timestamp": 123.5,
            "source_id": "cam1",
            "frame_id": 7,
            "class_id": 1,
            "class_name": "fire",
            "confidence": 0.91,
            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            "bbox_area": 4.0,
            "accepted": True,
            "reason": "accepted",
        }
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_detection_logger.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `fire_detection_alarm.logging.detection_logger`.

- [ ] **Step 3: Implement JSONL logger**

Create empty `fire_detection_alarm/logging/__init__.py`.

Create `fire_detection_alarm/logging/detection_logger.py`:

```python
import json
from pathlib import Path


class DetectionLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision):
        detection = decision.detection
        record = {
            "timestamp": decision.timestamp,
            "source_id": detection.source_id,
            "frame_id": detection.frame_id,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "bbox_xyxy": detection.bbox_xyxy,
            "bbox_area": detection.bbox_area,
            "accepted": decision.accepted,
            "reason": decision.reason,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_detection_logger.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
GIT_MASTER=1 git add fire_detection_alarm/logging/__init__.py fire_detection_alarm/logging/detection_logger.py tests/test_detection_logger.py
GIT_MASTER=1 git commit -m "feat: add detection decision logging"
```

---

### Task 5: Offline Demo Filtering Integration

**Files:**
- Modify: `configs/default.yaml`
- Modify: `scripts/demo_offline.py`

- [ ] **Step 1: Update config with detection log path**

Modify `configs/default.yaml` to include:

```yaml
logging:
  detections_path: "outputs/detections.jsonl"
```

- [ ] **Step 2: Update demo imports**

Modify `scripts/demo_offline.py` imports to include:

```python
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
```

- [ ] **Step 3: Initialize filtering components after config load**

In `scripts/demo_offline.py`, after `engine = YOLOEngine(...)`, add:

```python
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
```

- [ ] **Step 4: Replace detection rendering block with filter pipeline**

Replace this block:

```python
        detections = normalize_yolo_output(results[0], "demo", frame_id, time.time())
        frame = render_detections(frame, detections)
```

with:

```python
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
```

- [ ] **Step 5: Run existing tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_detection_filter.py tests/test_filtering.py tests/test_cooldown.py tests/test_detection_logger.py -q
```

Expected: PASS.

- [ ] **Step 6: Run full test suite**

Run:

```bash
./.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
GIT_MASTER=1 git add configs/default.yaml scripts/demo_offline.py
GIT_MASTER=1 git commit -m "feat: integrate filtering into offline demo"
```

---

### Task 6: Final Verification

**Files:**
- Verify all Milestone 2 files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
./.venv/bin/python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 2: Inspect git status**

Run:

```bash
GIT_MASTER=1 git status --short
```

Expected: clean working tree.

- [ ] **Step 3: Confirm Milestone 2 success criteria**

Verify these behaviors are covered by code and tests:

- low-confidence detections are rejected with `low_confidence`.
- disallowed classes are rejected with `class_not_allowed`.
- tiny detections are rejected with `bbox_too_small`.
- missing frame shape is rejected with `missing_frame_size`.
- single-frame detections are rejected with `not_persistent` in the demo integration.
- cooldown rejects repeated accepted detections with `cooldown_active`.
- JSONL logs include accepted and rejected reasons.
- offline demo renders accepted detections only.

- [ ] **Step 4: Commit any final plan checkbox updates only if the implementation changed the plan file**

If no plan file was edited during implementation, do not create a documentation-only cleanup commit.
