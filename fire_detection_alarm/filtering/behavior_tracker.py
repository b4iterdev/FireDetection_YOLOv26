from dataclasses import dataclass
from collections.abc import Sequence

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision


@dataclass
class _BehaviorTrack:
    first_area: float
    frames: int


class BehaviorTracker:
    min_track_frames: int
    max_stable_growth_ratio: float
    max_non_hazard_area_ratio: float
    min_growth_ratio: float
    tracks: dict[tuple[str, int], _BehaviorTrack]

    def __init__(
        self,
        min_track_frames: int = 3,
        max_stable_growth_ratio: float = 0.1,
        max_non_hazard_area_ratio: float = 0.01,
        min_growth_ratio: float = 0.5,
    ):
        self.min_track_frames = int(min_track_frames)
        self.max_stable_growth_ratio = float(max_stable_growth_ratio)
        self.max_non_hazard_area_ratio = float(max_non_hazard_area_ratio)
        self.min_growth_ratio = float(min_growth_ratio)
        self.tracks = {}

    def check(
        self,
        detections: Sequence[Detection],
        frame_shape: Sequence[int] | None,
    ) -> list[DetectionDecision]:
        if not detections:
            self.tracks.clear()
            return []

        frame_area = self._frame_area(frame_shape)
        active_keys: set[tuple[str, int]] = set()
        decisions: list[DetectionDecision] = []

        for detection in detections:
            key = (detection.source_id, detection.class_id)
            active_keys.add(key)
            track = self.tracks.get(key)
            if track is None:
                track = _BehaviorTrack(first_area=detection.bbox_area, frames=1)
                self.tracks[key] = track
            else:
                track.frames += 1

            decisions.append(self._decide(detection, track, frame_area))

        self.tracks = {key: value for key, value in self.tracks.items() if key in active_keys}
        return decisions

    def _decide(
        self,
        detection: Detection,
        track: _BehaviorTrack,
        frame_area: float,
    ) -> DetectionDecision:
        if track.frames < self.min_track_frames:
            return DetectionDecision(detection, True, "behavior_observed", detection.timestamp)

        growth_ratio = 0.0
        if track.first_area > 0:
            growth_ratio = (detection.bbox_area - track.first_area) / track.first_area

        if growth_ratio >= self.min_growth_ratio:
            return DetectionDecision(detection, True, "behavior_growing", detection.timestamp)

        area_ratio = detection.bbox_area / frame_area if frame_area > 0 else 0.0
        if (
            area_ratio <= self.max_non_hazard_area_ratio
            and growth_ratio <= self.max_stable_growth_ratio
        ):
            return DetectionDecision(detection, False, "stable_small_fire", detection.timestamp)

        return DetectionDecision(detection, True, "behavior_observed", detection.timestamp)

    @staticmethod
    def _frame_area(frame_shape: Sequence[int] | None) -> float:
        if frame_shape is None or len(frame_shape) < 2:
            return 0.0
        return float(frame_shape[0] * frame_shape[1])
