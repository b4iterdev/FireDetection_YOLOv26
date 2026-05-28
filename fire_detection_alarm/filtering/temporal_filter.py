class TemporalFilter:
    min_seconds: float
    min_frames: int

    def __init__(self, min_seconds: float = 2.0, min_frames: int = 0):
        self.min_seconds = float(min_seconds)
        self.min_frames = int(min_frames)
        self.active_starts: dict[str, float] = {}
        self.active_counts: dict[str, int] = {}

    def check(self, source_id: str, is_detected: bool, timestamp: float) -> bool:
        if not is_detected:
            _ = self.active_starts.pop(source_id, None)
            _ = self.active_counts.pop(source_id, None)
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
