from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

from fire_detection_alarm.filtering.decision import DetectionDecision


@dataclass
class LiveSummary:
    source_label: str
    source_type: str
    decisions: list[DetectionDecision]
    accepted_count: int
    frame_count: int
    processing_seconds: float
    completed_reason: str
    output_path: Path
    triggered_frame_paths: list[Path] = field(default_factory=list)
    input_path: Path | None = None

    @property
    def rejected_count(self) -> int:
        return len([decision for decision in self.decisions if not decision.accepted])

    @property
    def triggered_frame_count(self) -> int:
        return len(self.triggered_frame_paths)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(decision.reason for decision in self.decisions))

    @property
    def accepted_counts_by_class(self) -> dict[str, int]:
        return dict(
            Counter(
                decision.detection.class_name
                for decision in self.decisions
                if decision.accepted
            )
        )

    @property
    def max_confidence(self) -> float:
        confidences = [decision.detection.confidence for decision in self.decisions]
        if not confidences:
            return 0.0
        return max(confidences)

    @property
    def input_filename(self) -> str:
        if self.input_path is not None:
            return self.input_path.name
        return self.source_label

    @property
    def input_type(self) -> str:
        if self.source_type == "video_file":
            return "video"
        return self.source_type

    @property
    def output_available(self) -> bool:
        return self.output_path.exists()
