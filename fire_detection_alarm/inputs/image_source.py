import cv2
import numpy as np
from fire_detection_alarm.inputs.base import BaseSource

class ImageSource(BaseSource):
    def __init__(self, path):
        self.path = path
        self.frame = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frame is not None:
            return False, None
        
        self.frame = cv2.imread(self.path)
        if self.frame is not None:
            return True, self.frame
        return False, None

    def release(self):
        pass
