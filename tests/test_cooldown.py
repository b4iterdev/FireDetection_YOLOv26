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
