"""code for defining a relationship between object detection data and one or more behaviors of interest"""
import logging
import cv2
import numpy as np
import pandas as pd
from dbscan1d import DBSCAN1D

logger = logging.getLogger(__name__)


class BehaviorRecognizer:

    def __init__(self, config):
        self.config = config
        self.behavior_check_window = config.behavior_check_window
        self.min_individuals_roi = config.behavior_min_individuals_roi
        self.max_individuals_roi = config.behavior_max_individuals_roi
        self.use_dbscan = config.behavior_use_dbscan
        self.eps = config.behavior_max_gap_within_event
        self.min_samples = config.behavior_min_event_length // config.framegrab_interval
        self.data_buffer = []
        logger.debug('BehaviorRecognizer initialized')

    def append_data(self, timestamp, occupancy, thumbnail):
        self.data_buffer.append((timestamp, occupancy, thumbnail))
        while (len(self.data_buffer) >= 2) and (self.calc_buffer_length_seconds() > self.behavior_check_window):
            self.data_buffer.pop(0)

    def calc_activity_fraction(self):
        data = pd.DataFrame([x[:2] for x in self.data_buffer], columns=['timestamp', 'occupancy'])
        data_slice = data.query('@self.min_individuals_roi <= occupancy <= @self.max_individuals_roi')
        if self.use_dbscan:
            clustering_target = data_slice['timestamp'].values
            if clustering_target.size < self.min_samples:
                return 0.0
            labels = DBSCAN1D(self.eps, self.min_samples).fit_predict(clustering_target)
            labels = pd.Series(data=labels, index=data_slice.index).reindex(data.index, fill_value=-1)
            activity_length = 0
            for lid in labels.unique():
                if lid != -1:
                    start = data.timestamp.loc[labels[labels == lid].index.min()]
                    stop = data.timestamp.loc[labels[labels == lid].index.max()]
                    activity_length += (stop - start)
            activity_fraction = activity_length / self.calc_buffer_length_seconds()
            return activity_fraction
        else:
            return len(data_slice) / len(data)

    # def get_event_labels(self):
    #     data = pd.DataFrame([x[:2] for x in self.data_buffer], columns=['timestamp', 'occupancy'])
    #     data_slice = data.query('@self.min_individuals_roi <= occupancy <= @self.max_individuals_roi')
    #     clustering_target = data_slice['timestamp'].values
    #     if clustering_target.size < self.min_samples:
    #         return pd.Series(-1, index=data.index)
    #     labels = DBSCAN1D(self.eps, self.min_samples).fit_predict(clustering_target)
    #     labels = pd.Series(data=labels, index=data_slice.index).reindex(data.index, fill_value=-1)
    #     for lid in labels.unique():
    #         if lid != -1:
    #             labels.loc[labels[labels == lid].index.min():labels[labels == lid].index.max()] = lid
    #     return labels

    def thumbnails_to_mp4(self, output_path):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 1 // self.config.framegrab_interval
        height, width, _ = self.data_buffer[0][1].shape
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for thumbnail, _ in self.data_buffer:
            video.write(thumbnail)
        video.release()

    def calc_buffer_length_seconds(self):
        return self.data_buffer[-1][0] - self.data_buffer[0][0]

    def reset(self):
        self.data_buffer = []
