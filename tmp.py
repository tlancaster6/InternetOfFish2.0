from modules.data_collection import MockDataCollector
from modules.config_manager import ConfigManager
from modules.object_detection import DetectorBase


import pathlib
import sys
import numpy as np


FILE = pathlib.Path(__file__).resolve()
REPO_ROOT_DIR = FILE.parent  # repository root
MODEL_DIR = REPO_ROOT_DIR / 'models'
DEFAULT_DATA_DIR = REPO_ROOT_DIR / 'projects'
LOG_DIR = REPO_ROOT_DIR / 'logs'
TESTING_RESOURCEC_DIR = REPO_ROOT_DIR / 'resources'
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))
if not LOG_DIR.exists():
    LOG_DIR.mkdir()

config_path = DEFAULT_DATA_DIR / 'test_project' / 'config.yaml'
config = ConfigManager(config_path).config_as_namespace()
roi_detector = DetectorBase(MODEL_DIR / config.roi_model, config.roi_confidence_thresh)
ooi_detector = DetectorBase(MODEL_DIR / config.ooi_model, config.ooi_confidence_thresh)
collector = MockDataCollector(TESTING_RESOURCEC_DIR / 'sample_clip.mp4', config.framegrab_interval)

first_img = collector.capture_frame()
roi_det = roi_detector.detect(first_img)
roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax, roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]

