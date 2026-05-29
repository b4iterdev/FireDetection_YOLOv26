# Milestone 3 Live Stream Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add webcam and RTSP support to the existing demo CLI with bounded RTSP reconnects, live FPS throttling, and preview status overlays while preserving the existing YOLO → normalize → filter → log → render pipeline.

**Architecture:** Keep `scripts/demo_offline.py` as the single entrypoint and move live-input behavior into new source classes under `fire_detection_alarm/inputs/`. `WebcamSource` and `RTSPSource` will expose the same `read()` / `release()` contract as file-based sources, while `demo_offline.py` handles only source selection and preview rendering. RTSP reconnect state stays source-local; preview overlays stay script-local.

**Tech Stack:** Python 3.x, OpenCV (`cv2`), NumPy, PyYAML, Pytest, standard library (`time`, `dataclasses`, `typing`, `enum`).

---

## File Structure

### Create
- `fire_detection_alarm/inputs/live_status.py` — small shared status model for live sources and preview overlays.
- `fire_detection_alarm/inputs/webcam_source.py` — webcam capture wrapper with FPS throttling.
- `fire_detection_alarm/inputs/rtsp_source.py` — RTSP capture wrapper with bounded reconnects and FPS throttling.
- `tests/test_live_sources.py` — unit tests for live source behavior.
- `tests/test_demo_offline_source_selection.py` — integration-style tests for CLI source selection and preview-status integration.

### Modify
- `fire_detection_alarm/inputs/base.py` — add an optional status access point without breaking existing sources.
- `configs/default.yaml` — add `streaming` settings.
- `scripts/demo_offline.py` — source selection, preview overlay rendering, non-breaking live loop behavior.

### Keep Unchanged
- `fire_detection_alarm/models/yolo_engine.py`
- `fire_detection_alarm/detection/normalizer.py`
- `fire_detection_alarm/filtering/*`
- `fire_detection_alarm/logging/detection_logger.py`

---

### Task 1: Add live source status primitives

**Files:**
- Create: `fire_detection_alarm/inputs/live_status.py`
- Modify: `fire_detection_alarm/inputs/base.py`
- Test: `tests/test_live_sources.py`

- [ ] **Step 1: Write the failing tests**

```python
from fire_detection_alarm.inputs.base import BaseSource
from fire_detection_alarm.inputs.live_status import LiveSourceState, LiveSourceStatus


class DummySource(BaseSource):
    def read(self):
        return False, None

    def release(self):
        return None


def test_base_source_default_status_is_idle():
    source = DummySource()

    status = source.get_status()

    assert status.state is LiveSourceState.IDLE
    assert status.message == "idle"


def test_live_source_status_exposes_retry_fields():
    status = LiveSourceStatus(
        state=LiveSourceState.RECONNECTING,
        message="retrying rtsp stream",
        retry_attempts=2,
        retry_limit=5,
    )

    assert status.state is LiveSourceState.RECONNECTING
    assert status.retry_attempts == 2
    assert status.retry_limit == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_sources.py::test_base_source_default_status_is_idle -v`

Expected: FAIL with `ModuleNotFoundError` for `fire_detection_alarm.inputs.live_status` or `AttributeError` for missing `get_status`.

- [ ] **Step 3: Write minimal implementation**

```python
# fire_detection_alarm/inputs/live_status.py
from dataclasses import dataclass
from enum import Enum


class LiveSourceState(str, Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass(frozen=True)
class LiveSourceStatus:
    state: LiveSourceState = LiveSourceState.IDLE
    message: str = "idle"
    retry_attempts: int = 0
    retry_limit: int = 0
```

```python
# fire_detection_alarm/inputs/base.py
from abc import ABC, abstractmethod
import numpy as np

from fire_detection_alarm.inputs.live_status import LiveSourceStatus


class BaseSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def get_status(self) -> LiveSourceStatus:
        return LiveSourceStatus()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_live_sources.py -v`

Expected: PASS for the new status tests.

- [ ] **Step 5: Commit**

```bash
git add fire_detection_alarm/inputs/live_status.py fire_detection_alarm/inputs/base.py tests/test_live_sources.py
git commit -m "feat: add live source status model"
```

---

### Task 2: Add WebcamSource with FPS throttling

**Files:**
- Create: `fire_detection_alarm/inputs/webcam_source.py`
- Test: `tests/test_live_sources.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from fire_detection_alarm.inputs.webcam_source import WebcamSource
from fire_detection_alarm.inputs.live_status import LiveSourceState


class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_webcam_source_reads_frame(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "fire_detection_alarm.inputs.webcam_source.cv2.VideoCapture",
        lambda index: FakeCapture([frame]),
    )

    source = WebcamSource(0, max_fps=30)
    ret, read_frame = source.read()

    assert ret is True
    assert read_frame is not None
    assert source.get_status().state is LiveSourceState.STREAMING


def test_webcam_source_rejects_unopenable_camera(monkeypatch):
    class ClosedCapture(FakeCapture):
        def __init__(self):
            super().__init__([])

        def isOpened(self):
            return False

    monkeypatch.setattr(
        "fire_detection_alarm.inputs.webcam_source.cv2.VideoCapture",
        lambda index: ClosedCapture(),
    )

    try:
        WebcamSource(0, max_fps=30)
    except ValueError as exc:
        assert "Unable to open webcam" in str(exc)
    else:
        raise AssertionError("Expected webcam open failure")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_sources.py::test_webcam_source_reads_frame -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fire_detection_alarm.inputs.webcam_source'`.

- [ ] **Step 3: Write minimal implementation**

```python
import time

import cv2
import numpy as np

from fire_detection_alarm.inputs.base import BaseSource
from fire_detection_alarm.inputs.live_status import LiveSourceState, LiveSourceStatus


class WebcamSource(BaseSource):
    def __init__(self, index: int, max_fps: float):
        self.index = int(index)
        self.max_fps = float(max_fps)
        self.min_interval = 0.0 if self.max_fps <= 0 else 1.0 / self.max_fps
        self.last_frame_time = 0.0
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open webcam index {self.index}")
        self._status = LiveSourceStatus(
            state=LiveSourceState.STREAMING,
            message=f"webcam {self.index} streaming",
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._status = LiveSourceStatus(
                    state=LiveSourceState.FAILED,
                    message=f"webcam {self.index} read failed",
                )
                return False, None

            now = time.monotonic()
            if self.min_interval <= 0 or now - self.last_frame_time >= self.min_interval:
                self.last_frame_time = now
                self._status = LiveSourceStatus(
                    state=LiveSourceState.STREAMING,
                    message=f"webcam {self.index} streaming",
                )
                return True, frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    def get_status(self) -> LiveSourceStatus:
        return self._status
```

- [ ] **Step 4: Add FPS throttling test before refactoring**

```python
def test_webcam_source_drops_extra_frames(monkeypatch):
    frames = [
        np.full((2, 2, 3), 1, dtype=np.uint8),
        np.full((2, 2, 3), 2, dtype=np.uint8),
        np.full((2, 2, 3), 3, dtype=np.uint8),
    ]
    times = iter([0.00, 0.01, 0.02, 0.60])

    monkeypatch.setattr(
        "fire_detection_alarm.inputs.webcam_source.cv2.VideoCapture",
        lambda index: FakeCapture(frames),
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.webcam_source.time.monotonic",
        lambda: next(times),
    )

    source = WebcamSource(0, max_fps=2)
    ret, frame = source.read()

    assert ret is True
    assert int(frame[0, 0, 0]) == 3
```

- [ ] **Step 5: Make the FPS test pass with minimal code**

```python
def read(self) -> tuple[bool, np.ndarray | None]:
    latest_frame = None
    while True:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._status = LiveSourceStatus(
                state=LiveSourceState.FAILED,
                message=f"webcam {self.index} read failed",
            )
            return False, None

        latest_frame = frame
        now = time.monotonic()
        if self.min_interval <= 0 or now - self.last_frame_time >= self.min_interval:
            self.last_frame_time = now
            self._status = LiveSourceStatus(
                state=LiveSourceState.STREAMING,
                message=f"webcam {self.index} streaming",
            )
            return True, latest_frame
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_live_sources.py -v`

Expected: PASS for webcam open/read/failure/FPS tests.

- [ ] **Step 7: Commit**

```bash
git add fire_detection_alarm/inputs/webcam_source.py tests/test_live_sources.py
git commit -m "feat: add webcam source"
```

---

### Task 3: Add RTSPSource with bounded reconnects

**Files:**
- Create: `fire_detection_alarm/inputs/rtsp_source.py`
- Test: `tests/test_live_sources.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from fire_detection_alarm.inputs.live_status import LiveSourceState
from fire_detection_alarm.inputs.rtsp_source import RTSPSource


class SequencedCapture:
    def __init__(self, opened, reads):
        self._opened = opened
        self._reads = list(reads)
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._reads:
            return self._reads.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_rtsp_source_retries_after_read_failure(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    captures = iter([
        SequencedCapture(True, [(False, None)]),
        SequencedCapture(True, [(True, frame)]),
    ])

    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.cv2.VideoCapture",
        lambda url: next(captures),
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.time.monotonic",
        lambda: 10.0,
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.time.sleep",
        lambda seconds: None,
    )

    source = RTSPSource(
        "rtsp://camera/stream",
        max_fps=5,
        retry_limit=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        backoff_multiplier=2.0,
    )
    ret, read_frame = source.read()

    assert ret is True
    assert read_frame is not None
    assert source.get_status().state is LiveSourceState.STREAMING


def test_rtsp_source_marks_failed_after_retry_budget(monkeypatch):
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.cv2.VideoCapture",
        lambda url: SequencedCapture(False, []),
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.time.sleep",
        lambda seconds: None,
    )

    source = RTSPSource(
        "rtsp://camera/stream",
        max_fps=5,
        retry_limit=2,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        backoff_multiplier=2.0,
    )
    ret, frame = source.read()

    assert ret is False
    assert frame is None
    assert source.get_status().state is LiveSourceState.FAILED
    assert source.get_status().retry_attempts == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_sources.py::test_rtsp_source_retries_after_read_failure -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fire_detection_alarm.inputs.rtsp_source'`.

- [ ] **Step 3: Write minimal implementation**

```python
import time

import cv2
import numpy as np

from fire_detection_alarm.inputs.base import BaseSource
from fire_detection_alarm.inputs.live_status import LiveSourceState, LiveSourceStatus


class RTSPSource(BaseSource):
    def __init__(
        self,
        url: str,
        max_fps: float,
        retry_limit: int,
        initial_backoff_seconds: float,
        max_backoff_seconds: float,
        backoff_multiplier: float,
    ):
        self.url = url
        self.max_fps = float(max_fps)
        self.min_interval = 0.0 if self.max_fps <= 0 else 1.0 / self.max_fps
        self.retry_limit = int(retry_limit)
        self.initial_backoff_seconds = float(initial_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.backoff_multiplier = float(backoff_multiplier)
        self.last_frame_time = 0.0
        self.last_frame = None
        self.retry_attempts = 0
        self.cap = None
        self._status = LiveSourceStatus(message="connecting rtsp stream")
        self._open_capture()

    def _open_capture(self) -> bool:
        self.cap = cv2.VideoCapture(self.url)
        return self.cap.isOpened()

    def _mark_reconnecting(self) -> None:
        self._status = LiveSourceStatus(
            state=LiveSourceState.RECONNECTING,
            message="retrying rtsp stream",
            retry_attempts=self.retry_attempts,
            retry_limit=self.retry_limit,
        )

    def _mark_failed(self) -> None:
        self._status = LiveSourceStatus(
            state=LiveSourceState.FAILED,
            message="rtsp stream failed",
            retry_attempts=self.retry_attempts,
            retry_limit=self.retry_limit,
        )

    def _reconnect(self) -> bool:
        while self.retry_attempts < self.retry_limit:
            self.retry_attempts += 1
            self._mark_reconnecting()
            backoff = min(
                self.max_backoff_seconds,
                self.initial_backoff_seconds * (self.backoff_multiplier ** (self.retry_attempts - 1)),
            )
            time.sleep(backoff)
            if self.cap is not None:
                self.cap.release()
            if self._open_capture():
                self._status = LiveSourceStatus(
                    state=LiveSourceState.STREAMING,
                    message="rtsp stream connected",
                    retry_attempts=self.retry_attempts,
                    retry_limit=self.retry_limit,
                )
                return True

        self._mark_failed()
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        latest_frame = None
        while True:
            if self.cap is None or not self.cap.isOpened():
                if not self._reconnect():
                    return False, None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self._reconnect():
                    return False, None
                continue

            latest_frame = frame
            now = time.monotonic()
            if self.min_interval <= 0 or now - self.last_frame_time >= self.min_interval:
                self.last_frame_time = now
                self.last_frame = latest_frame
                self._status = LiveSourceStatus(
                    state=LiveSourceState.STREAMING,
                    message="rtsp stream connected",
                    retry_attempts=self.retry_attempts,
                    retry_limit=self.retry_limit,
                )
                return True, latest_frame

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def get_status(self) -> LiveSourceStatus:
        return self._status
```

- [ ] **Step 4: Add FPS throttling RTSP test**

```python
def test_rtsp_source_returns_newest_eligible_frame(monkeypatch):
    frames = [
        np.full((2, 2, 3), 1, dtype=np.uint8),
        np.full((2, 2, 3), 2, dtype=np.uint8),
        np.full((2, 2, 3), 3, dtype=np.uint8),
    ]
    times = iter([0.00, 0.01, 0.02, 0.60])

    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.cv2.VideoCapture",
        lambda url: SequencedCapture(True, [(True, frame) for frame in frames]),
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "fire_detection_alarm.inputs.rtsp_source.time.sleep",
        lambda seconds: None,
    )

    source = RTSPSource(
        "rtsp://camera/stream",
        max_fps=2,
        retry_limit=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        backoff_multiplier=2.0,
    )
    ret, frame = source.read()

    assert ret is True
    assert int(frame[0, 0, 0]) == 3
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_live_sources.py -v`

Expected: PASS for RTSP reconnect/budget/FPS tests along with existing live-source tests.

- [ ] **Step 6: Commit**

```bash
git add fire_detection_alarm/inputs/rtsp_source.py tests/test_live_sources.py
git commit -m "feat: add rtsp source reconnect handling"
```

---

### Task 4: Wire live source selection and preview overlays into demo_offline

**Files:**
- Modify: `scripts/demo_offline.py`
- Modify: `configs/default.yaml`
- Test: `tests/test_demo_offline_source_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts import demo_offline


def test_select_source_uses_webcam_for_numeric_input(monkeypatch):
    monkeypatch.setattr(demo_offline, "WebcamSource", lambda index, max_fps: ("webcam", index, max_fps))
    monkeypatch.setattr(demo_offline, "RTSPSource", object())
    monkeypatch.setattr(demo_offline, "ImageSource", object())
    monkeypatch.setattr(demo_offline, "VideoSource", lambda path: ("video", path))

    source = demo_offline.create_source("0", {"inference": {"max_fps": 5}, "streaming": {}})

    assert source == ("webcam", 0, 5)


def test_select_source_uses_rtsp_for_rtsp_url(monkeypatch):
    cfg = {
        "inference": {"max_fps": 5},
        "streaming": {
            "retry_limit": 5,
            "initial_backoff_seconds": 1.0,
            "max_backoff_seconds": 8.0,
            "backoff_multiplier": 2.0,
        },
    }
    monkeypatch.setattr(demo_offline, "RTSPSource", lambda **kwargs: kwargs)
    monkeypatch.setattr(demo_offline, "WebcamSource", object())
    monkeypatch.setattr(demo_offline, "ImageSource", object())
    monkeypatch.setattr(demo_offline, "VideoSource", object())

    source = demo_offline.create_source("rtsp://camera/stream", cfg)

    assert source["url"] == "rtsp://camera/stream"
    assert source["retry_limit"] == 5


def test_overlay_status_adds_text(monkeypatch):
    calls = []

    monkeypatch.setattr(demo_offline.cv2, "putText", lambda *args, **kwargs: calls.append(args) or args[0])

    frame = object()
    status = demo_offline.LiveSourceStatus(
        state=demo_offline.LiveSourceState.RECONNECTING,
        message="retrying rtsp stream",
        retry_attempts=2,
        retry_limit=5,
    )

    returned = demo_offline.overlay_live_status(frame, status)

    assert returned is frame
    assert calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_offline_source_selection.py -v`

Expected: FAIL because `create_source` and `overlay_live_status` do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# configs/default.yaml
streaming:
  retry_limit: 5
  initial_backoff_seconds: 1.0
  max_backoff_seconds: 8.0
  backoff_multiplier: 2.0
```

```python
# scripts/demo_offline.py
from fire_detection_alarm.inputs.live_status import LiveSourceState, LiveSourceStatus
from fire_detection_alarm.inputs.rtsp_source import RTSPSource
from fire_detection_alarm.inputs.webcam_source import WebcamSource


def parse_webcam_index(input_value: str) -> int:
    if input_value.isdigit():
        return int(input_value)
    if input_value.startswith("webcam://"):
        return int(input_value.split("://", 1)[1])
    raise ValueError(f"Unsupported webcam input: {input_value}")


def create_source(input_value: str, cfg: dict):
    max_fps = cfg["inference"]["max_fps"]
    streaming_cfg = cfg.get("streaming", {})
    lower_value = input_value.lower()
    if lower_value.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return ImageSource(input_value)
    if lower_value.startswith("rtsp://"):
        return RTSPSource(
            url=input_value,
            max_fps=max_fps,
            retry_limit=streaming_cfg["retry_limit"],
            initial_backoff_seconds=streaming_cfg["initial_backoff_seconds"],
            max_backoff_seconds=streaming_cfg["max_backoff_seconds"],
            backoff_multiplier=streaming_cfg["backoff_multiplier"],
        )
    if input_value.isdigit() or lower_value.startswith("webcam://"):
        return WebcamSource(parse_webcam_index(input_value), max_fps=max_fps)
    return VideoSource(input_value)


def overlay_live_status(frame, status: LiveSourceStatus):
    if status.state is LiveSourceState.STREAMING:
        return frame
    text = status.message
    if status.retry_limit:
        text = f"{text} ({status.retry_attempts}/{status.retry_limit})"
    return cv2.putText(
        frame,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
```

- [ ] **Step 4: Update the main loop with minimal live behavior**

```python
source = create_source(args.input, cfg)

while True:
    ret, frame = source.read()
    if not ret or frame is None:
        status = source.get_status()
        if status.state in {LiveSourceState.RECONNECTING, LiveSourceState.FAILED}:
            last_frame = getattr(source, "last_frame", None)
            if last_frame is not None:
                preview_frame = overlay_live_status(last_frame.copy(), status)
                cv2.imshow("Fire Detection Demo", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
        break

    last_frame = frame.copy()
    # existing inference/filter/render pipeline remains here
    rendered = render_detections(frame, accepted_detections)
    rendered = overlay_live_status(rendered, source.get_status())
    cv2.imshow("Fire Detection Demo", rendered)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_demo_offline_source_selection.py -v`

Expected: PASS for source selection and overlay tests.

- [ ] **Step 6: Run targeted regression tests**

Run: `pytest tests/test_live_sources.py tests/test_demo_offline_source_selection.py tests/test_demo_offline_import.py -v`

Expected: PASS for all live-source and demo-entrypoint tests.

- [ ] **Step 7: Commit**

```bash
git add scripts/demo_offline.py configs/default.yaml tests/test_demo_offline_source_selection.py
git commit -m "feat: add live stream support to demo entrypoint"
```

---

### Task 5: Final verification and cleanup

**Files:**
- Verify only: `fire_detection_alarm/inputs/base.py`
- Verify only: `fire_detection_alarm/inputs/live_status.py`
- Verify only: `fire_detection_alarm/inputs/webcam_source.py`
- Verify only: `fire_detection_alarm/inputs/rtsp_source.py`
- Verify only: `scripts/demo_offline.py`
- Verify only: `configs/default.yaml`
- Verify only: `tests/test_live_sources.py`
- Verify only: `tests/test_demo_offline_source_selection.py`

- [ ] **Step 1: Run the Milestone 3-focused test suite**

Run: `pytest tests/test_live_sources.py tests/test_demo_offline_source_selection.py tests/test_demo_offline_import.py tests/test_sources.py -v`

Expected: PASS.

- [ ] **Step 2: Run the full project test suite**

Run: `pytest -q`

Expected: PASS with the project’s existing suite plus the new Milestone 3 coverage.

- [ ] **Step 3: Run diagnostics on changed files**

Run diagnostics for:

- `fire_detection_alarm/inputs/base.py`
- `fire_detection_alarm/inputs/live_status.py`
- `fire_detection_alarm/inputs/webcam_source.py`
- `fire_detection_alarm/inputs/rtsp_source.py`
- `scripts/demo_offline.py`
- `tests/test_live_sources.py`
- `tests/test_demo_offline_source_selection.py`

Expected: no new errors.

- [ ] **Step 4: Manually smoke-test CLI shape**

Run:

```bash
python scripts/demo_offline.py --help
```

Expected: help output renders successfully and still documents `--input` and `--model`.

- [ ] **Step 5: Commit any final cleanup**

```bash
git add fire_detection_alarm/inputs/base.py fire_detection_alarm/inputs/live_status.py fire_detection_alarm/inputs/webcam_source.py fire_detection_alarm/inputs/rtsp_source.py scripts/demo_offline.py configs/default.yaml tests/test_live_sources.py tests/test_demo_offline_source_selection.py
git commit -m "test: cover live stream source behavior"
```

Only create this commit if verification required a real code change. If no cleanup was needed, skip this step.

---

## Self-Review

### Spec coverage
- Webcam input support: Task 2 + Task 4
- RTSP input support: Task 3 + Task 4
- Retry budget of 5 by default: Task 3 + Task 4 config wiring
- Keep process alive after retry exhaustion: Task 3 + Task 4
- Preview status overlay: Task 4
- FPS dropping/newest eligible frame: Task 2 + Task 3
- Tests for live behavior: Tasks 1–5

### Placeholder scan
- No `TBD`, `TODO`, or “implement later” placeholders remain.
- Each task includes exact file paths, explicit tests, commands, and commit boundaries.

### Type consistency
- Shared status types are `LiveSourceState` and `LiveSourceStatus` across all tasks.
- Source interface remains `read()` / `release()` / `get_status()`.
- Demo script helper names stay consistent: `parse_webcam_index`, `create_source`, `overlay_live_status`.
