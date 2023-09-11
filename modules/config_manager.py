"""code for reading and writing project parameters"""

import yaml
from types import SimpleNamespace
import logging
import pathlib
logger = logging.getLogger(__name__)

class ConfigManager:

    def __init__(self, config_path: pathlib.Path):
        """
        class for reading and writing project configuration files
        :param config_path: path to config file

        description of config.yaml parameters:

        project_id: unique name for project

        """
        self.config_path = config_path
        if config_path.exists():
            self.load_config()
        else:
            self.config = None
        logger.debug('ConfigManager initialized')

    def load_config(self):
        with open(str(self.config_path), 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
        logger.debug('config loaded')

    def write_config(self):
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        with open(str(self.config_path), 'w') as f:
            yaml.dump(self.config, f)
        logger.debug(f'config written to {self.config_path}')

    def generate_new_config(self):
        config = {
            'project_id': self.config_path.parent.name,
            'cloud_data_dir': None,   # cloud path, including the rclone remote, where the project will be stored
            'user_email': None,
            'min_notification_interval': 600,
            'max_notifications_per_day': 20,
            'roi_model': 'roi.tflite',
            'ooi_model': 'ooi.tflite',
            'roi_confidence_thresh': 0.75,
            'ooi_confidence_thresh': 0.25,
            'behavior_check_window': 60,
            'behavior_check_interval': 30,  # seconds between behavior checks
            'behavior_min_individuals_roi': 2,   # min number of individuals in ROI during behavior event
            'behavior_max_individuals_roi': 2,   # max number of individuals in ROI during behavior event
            'framerate': 30,
            'h_resolution': 1280,
            'v_resolution': 720,
            'framegrab_interval': 0.2,
            'roi_update_interval': 600,
            'start_hour': 7,
            'end_hour': 19,
            'video_split_hours': 3,
            'test': False
            }
        self.config = config
        logger.debug('new config generated')
        self.write_config()

    def config_as_namespace(self):
        return SimpleNamespace(**self.config)

