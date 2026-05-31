from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.behavior_tracker import BehaviorTracker


def make_detection(frame_id: int, bbox_area: float, timestamp: float = 100.0) -> Detection:
    return Detection(
        source_id="cam1",
        frame_id=frame_id,
        timestamp=timestamp,
        class_id=1,
        class_name="fire",
        confidence=0.9,
        bbox_xyxy=[10.0, 10.0, 110.0, 110.0],
        bbox_area=bbox_area,
    )


def test_behavior_tracker_rejects_stable_small_fire():
    tracker = BehaviorTracker(
        min_track_frames=3,
        max_stable_growth_ratio=0.1,
        max_non_hazard_area_ratio=0.01,
        min_growth_ratio=0.5,
    )
    frame_shape = (1000, 1000, 3)

    assert tracker.check([make_detection(1, 3000.0, 100.0)], frame_shape)[0].accepted is True
    assert tracker.check([make_detection(2, 3050.0, 101.0)], frame_shape)[0].accepted is True
    decision = tracker.check([make_detection(3, 3100.0, 102.0)], frame_shape)[0]

    assert decision.accepted is False
    assert decision.reason == "stable_small_fire"


def test_behavior_tracker_accepts_growing_fire():
    tracker = BehaviorTracker(
        min_track_frames=3,
        max_stable_growth_ratio=0.1,
        max_non_hazard_area_ratio=0.01,
        min_growth_ratio=0.5,
    )
    frame_shape = (1000, 1000, 3)

    tracker.check([make_detection(1, 3000.0, 100.0)], frame_shape)
    tracker.check([make_detection(2, 4500.0, 101.0)], frame_shape)
    decision = tracker.check([make_detection(3, 6000.0, 102.0)], frame_shape)[0]

    assert decision.accepted is True
    assert decision.reason == "behavior_growing"


def test_behavior_tracker_resets_when_detection_disappears():
    tracker = BehaviorTracker(min_track_frames=2)
    frame_shape = (1000, 1000, 3)

    tracker.check([make_detection(1, 3000.0, 100.0)], frame_shape)
    tracker.check([], frame_shape)
    decision = tracker.check([make_detection(2, 3000.0, 102.0)], frame_shape)[0]

    assert decision.accepted is True
    assert decision.reason == "behavior_observed"
