from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    source_id: str
    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]
    bbox_area: float
