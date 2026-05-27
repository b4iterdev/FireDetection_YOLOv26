class TemporalFilter:
    def __init__(self, min_seconds=2.0):
        self.min_seconds = min_seconds
        self.active_starts = {}

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
