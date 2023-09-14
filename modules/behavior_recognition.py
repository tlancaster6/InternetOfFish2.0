"""code for defining a relationship between object detection data and one or more behaviors of interest"""
import logging
import cv2
logger = logging.getLogger(__name__)

class BehaviorRecognizer:

    def __init__(self, config):
        self.config = config
        self.behavior_check_window = config.behavior_check_window
        self.data_buffer_length = self.behavior_check_window // config.framegrab_interval
        self.min_individuals_roi = config.behavior_min_individuals_roi
        self.max_individuals_roi = config.behavior_max_individuals_roi
        self.data_buffer = []
        logger.debug('BehaviorRecognizer initialized')

    def append_data(self, occupancy, thumbnail):
        self.data_buffer.append((occupancy, thumbnail))
        if len(self.data_buffer) > self.data_buffer_length:
            self.data_buffer.pop(0)

    def check_for_behavior(self):
        if len(self.data_buffer) < self.behavior_check_window:
            return

    def thumbnails_to_mp4(self, output_path):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 1 // self.config.framegrab_interval
        height, width, _ = self.data_buffer[0][1].shape
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for thumbnail, _ in self.data_buffer:
            video.write(thumbnail)
        video.release()


