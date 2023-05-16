"""code for collecting basic video data and storing it locally on the raspberry pi"""

import picamera
import numpy as np
import datetime
import cv2


class DataCollector:

    def __init__(self, video_dir, picamera_kwargs=None):
        self.picamera_kwargs = picamera_kwargs
        self.video_dir = video_dir
        self.cam = self.init_camera(picamera_kwargs)
        self.resolution = self.cam.resolution

    def init_camera(self, picamera_kwargs):
        if picamera_kwargs:
            return picamera.PiCamera(**picamera_kwargs)
        return picamera.PiCamera()

    def generate_h264_path(self):
        iso_string = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '_')
        return self.video_dir / f'{iso_string}.h264'

    def start_recording(self):
        self.cam.start_recording(self.generate_h264_path())

    def split_recording(self):
        self.cam.split_recording(self.generate_h264_path())

    def stop_recording(self):
        self.cam.stop_recording()

    def capture_frame(self):
        image = np.empty((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        self.cam.capture(image, format='rgb', use_video_port=True)
        return image

    def shutdown(self):
        self.cam.close()

