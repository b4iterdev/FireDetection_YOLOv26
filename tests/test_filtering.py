import pytest
from fire_detection_alarm.filtering.temporal_filter import TemporalFilter

def test_temporal_persistence():
    f = TemporalFilter(min_seconds=1.0)
    
    assert f.check("cam1", True, timestamp=100.0) is False
    assert f.check("cam1", True, timestamp=100.5) is False
    assert f.check("cam1", True, timestamp=101.1) is True
    assert f.check("cam1", False, timestamp=101.2) is False
    assert f.check("cam1", True, timestamp=101.3) is False
