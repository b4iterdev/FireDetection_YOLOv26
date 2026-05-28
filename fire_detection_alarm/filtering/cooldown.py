class CooldownTracker:
    cooldown_seconds: float
    last_accepted_at: dict[str, float]

    def __init__(self, cooldown_seconds: float = 60) -> None:
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_accepted_at = {}

    def check(self, source_id: str, timestamp: float) -> bool:
        last_accepted = self.last_accepted_at.get(source_id)
        if last_accepted is None:
            self.last_accepted_at[source_id] = timestamp
            return True

        if timestamp - last_accepted >= self.cooldown_seconds:
            self.last_accepted_at[source_id] = timestamp
            return True

        return False
