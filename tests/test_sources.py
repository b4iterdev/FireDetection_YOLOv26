import pytest
import numpy as np
import cv2
import os
from fire_detection_alarm.inputs.image_source import ImageSource
from fire_detection_alarm.inputs.video_source import VideoSource

def test_image_source(tmp_path):
    img_path = str(tmp_path / "test.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    
    source = ImageSource(img_path)
    ret, frame = source.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (100, 100, 3)
    
    ret, frame = source.read()
    assert ret is False
    assert frame is None

def test_video_source_release():
    source = VideoSource("dummy_path.mp4")
    source.release()
