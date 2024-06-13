"""code for collecting basic video data and storing it locally on the raspberry pi"""

import picamera
import numpy as np
import datetime
import cv2
import logging
logger = logging.getLogger(__name__)
from time import sleep

class DataCollector:

    def __init__(self, video_dir, picamera_kwargs=None):
        self.picamera_kwargs = picamera_kwargs
        self.video_dir = video_dir
        self.video_dir.mkdir(exist_ok=True, parents=True)
        self.cam = self.init_camera(picamera_kwargs)
        self.resolution = self.cam.resolution
        logger.debug('DataCollector initialized')

    def init_camera(self, picamera_kwargs):
        if picamera_kwargs:
            cam = picamera.PiCamera(**picamera_kwargs)
        else:
            cam = picamera.PiCamera()
        logger.info('camera initialized')
        return cam

    def generate_h264_path(self):
        iso_string = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '_')
        return self.video_dir / f'{iso_string}.h264'

    def start_recording(self):
        self.cam.start_recording(str(self.generate_h264_path()))
        sleep(2)
        logger.info('recording started')

    def split_recording(self):
        self.cam.split_recording(str(self.generate_h264_path()))
        logger.info('recording split')

    def stop_recording(self):
        self.cam.stop_recording()
        logger.info('recording stopped')

    def capture_frame(self):
        image = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        self.cam.capture(image, format='rgb', use_video_port=True)
        return image

    def shutdown(self):
        try:
            self.stop_recording()
        except picamera.PiCameraNotRecording:
            logger.debug('Could not stop recording because camera was not recording. Skipping.')
        self.cam.close()
        logger.debug('DataCollector shutdown complete')
