import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fire_detection_alarm.filtering.decision import DetectionDecision
import scripts.demo_offline as demo_offline


def test_demo_offline_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/demo_offline.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_demo_offline_video_persistence_uses_media_time_while_records_keep_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions: list[DetectionDecision] = []

    class FakeEngine:
        def __init__(self, model_path: str, device: str):
            self.model_path: str = model_path
            self.device: str = device

        def predict(self, frame: np.ndarray, conf: float, iou: float) -> list[SimpleNamespace]:
            _ = (frame, conf, iou)
            boxes = SimpleNamespace(data=np.array([[0.0, 0.0, 4.0, 4.0, 0.9, 1.0]]))
            return [SimpleNamespace(boxes=boxes, names={0: "smoke", 1: "fire"})]

    class FakeVideoSource:
        frames: list[np.ndarray] = [
            np.full((4, 6, 3), 1, dtype=np.uint8),
            np.full((4, 6, 3), 2, dtype=np.uint8),
            np.full((4, 6, 3), 3, dtype=np.uint8),
        ]
        media_times: list[float] = [0.0, 1.1, 2.2]

        def __init__(self, path: str):
            self.path: str = path
            self.index: int = 0
            self.released: bool = False

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()

        def media_time_seconds(self, frame_id: int) -> float:
            return self.media_times[frame_id]

        def release(self) -> None:
            self.released = True

    class CapturingLogger:
        def __init__(self, log_path: str):
            self.log_path: str = log_path

        def write(self, decision: DetectionDecision) -> None:
            decisions.append(decision)

    config = {
        "model": {"device": "cpu"},
        "classes": {"allowed": [0, 1]},
        "inference": {"confidence_threshold": 0.25, "iou_threshold": 0.45},
        "filtering": {
            "min_bbox_area_ratio": 0.0,
            "min_persistence_seconds": 2.0,
            "min_consecutive_frames": 3,
        },
        "behavior_tracking": {
            "min_track_frames": 1,
            "max_stable_growth_ratio": 1.0,
            "max_non_hazard_area_ratio": 1.0,
            "min_growth_ratio": 0.0,
        },
    }
    wall_times = iter([1000.0, 1000.1, 1000.2])

    monkeypatch.setattr(sys, "argv", ["demo_offline.py", "--input", "sample.mp4", "--model", "model.pt"])
    monkeypatch.setattr(demo_offline, "load_config", lambda: config)
    monkeypatch.setattr(demo_offline, "YOLOEngine", FakeEngine)
    monkeypatch.setattr(demo_offline, "VideoSource", FakeVideoSource)
    monkeypatch.setattr(demo_offline, "DetectionLogger", CapturingLogger)
    monkeypatch.setattr("scripts.demo_offline.time.time", lambda: next(wall_times))
    monkeypatch.setattr("scripts.demo_offline.render_detections", lambda frame, detections: frame)
    monkeypatch.setattr("scripts.demo_offline.cv2.imshow", lambda window, frame: None)
    monkeypatch.setattr("scripts.demo_offline.cv2.waitKey", lambda delay: -1)
    monkeypatch.setattr("scripts.demo_offline.cv2.destroyAllWindows", lambda: None)

    demo_offline.main()

    assert [decision.reason for decision in decisions] == [
        "not_persistent",
        "not_persistent",
        "accepted",
    ]
    assert [decision.timestamp for decision in decisions] == [1000.0, 1000.1, 1000.2]
    assert [decision.detection.timestamp for decision in decisions] == [1000.0, 1000.1, 1000.2]
