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
