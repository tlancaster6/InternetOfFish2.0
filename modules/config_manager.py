"""code for reading and writing project parameters"""

import yaml
from types import SimpleNamespace

class ConfigManager:

    def __init__(self, config_path):
        self.config_path = config_path
        if config_path.exists():
            self.load_config()
        else:
            self.config = None

    def load_config(self):
        with open(str(self.config_path), 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)

    def write_config(self):
        with open(str(self.config_path), 'w') as f:
            yaml.dump(self.config, f)

    def generate_new_config(self, project_id):
        config = {
            'project_id': project_id,
            'cloud_project_dir': None,   # cloud path, including the rclone remote, where the project will be stored
            'roi_model': 'roi.tflite',
            'ooi_model': 'ooi.tflite',
            'roi_confidence_thresh': 0.75,
            'ooi_confidence_thresh': 0.25,
            'framerate': 30,
            'h_resolution': 900,
            'v_resolution': 720,
            'framegrab_interval': 0.2,
            'roi_update_interval': 600,
            'start_hour': 7,
            'end_hour': 19,
            'video_split_hours': 3,
            'test': False
            }
        self.config = config
        self.write_config()

    def config_as_namespace(self):
        return SimpleNamespace(**self.config)

