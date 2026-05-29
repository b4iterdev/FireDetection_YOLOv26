from typing import Sequence
import numpy as np

from fire_detection_alarm.detection.schema import Detection
from fire_detection_alarm.filtering.decision import DetectionDecision


class FalseNetFilter:
    """
    A mock filter to simulate FalseNet's role of classifying whether a
    fire detection is a real threat or a non-threatening event (e.g., lighter).
    """

    def __init__(self, threat_threshold: float = 0.5):
        # In a real implementation, this would load a model.
        # For this mock, we'll use a simple heuristic based on bbox size.
        self.threat_threshold = threat_threshold

    def classify(self, frame: np.ndarray, detections: Sequence[Detection]) -> list[DetectionDecision]:
        """
        Classifies a list of detections.

        Args:
            frame: The full video frame.
            detections: A list of `Detection` objects to classify.

        Returns:
            A list of `DetectionDecision` objects.
        """
        decisions = []
        for detection in detections:
            # Mock logic: Assume smaller fires are non-threats (lighters, candles).
            # A real FalseNet would use a neural network on the cropped bbox.
            bbox_area = detection.bbox_area
            frame_area = frame.shape[0] * frame.shape[1]
            
            # Heuristic: if bbox is less than 0.5% of frame area, it's a "non-threat".
            # This is a stand-in for a real model's confidence score.
            mock_threat_score = 1.0 if (bbox_area / frame_area) > 0.005 else 0.1

            if mock_threat_score >= self.threat_threshold:
                decision = DetectionDecision(
                    detection, accepted=True, reason="falsenet_threat", timestamp=detection.timestamp
                )
            else:
                decision = DetectionDecision(
                    detection, accepted=False, reason="falsenet_non_threat", timestamp=detection.timestamp
                )
            decisions.append(decision)
        return decisions
