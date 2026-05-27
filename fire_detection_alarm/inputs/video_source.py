import cv2
from fire_detection_alarm.inputs.base import BaseSource

class VideoSource(BaseSource):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
