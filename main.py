import pathlib
import sys
import argparse
from datetime import datetime, timedelta, time
import numpy as np
import pause
import logging
import cv2
from logging.handlers import RotatingFileHandler

from modules.data_collection import DataCollector
from modules.object_detection import DetectorBase
from modules.upload_automation import Uploader
from modules.behavior_recognition import BehaviorRecognizer
from modules.config_manager import ConfigManager
from modules.email_notification import Notifier, Notification


# establish filesystem locations
FILE = pathlib.Path(__file__).resolve()
REPO_ROOT_DIR = FILE.parent  # repository root
MODEL_DIR = REPO_ROOT_DIR / 'models'
DEFAULT_DATA_DIR = REPO_ROOT_DIR / 'projects'
LOG_DIR = REPO_ROOT_DIR / 'logs'
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))
if not LOG_DIR.exists():
    LOG_DIR.mkdir()

# initiate logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(name)-16s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = RotatingFileHandler(str(LOG_DIR / 'debug.log'), maxBytes=500000, backupCount=1)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def new_project(config_path):
    config_manager = ConfigManager(config_path)
    config_manager.generate_new_config()
    print(f'new project config generated and saved to {config_path}. '
          'Edit this file if desired, then re-run main.py to initiate data collection. Note that, to enable email '
          'notifications and rclone uploads, you must supply the "cloud_data_dir" and "user_email" fields manually '
          'in the config ')


class Runner:

    def __init__(self, config_path: pathlib.Path):
        self.project_dir = config_path.parent
        self.video_dir = self.project_dir / 'Videos'
        self.config = ConfigManager(config_path).config_as_namespace()
        self.start_time = time(hour=self.config.start_hour)
        self.end_time = time(hour=self.config.end_hour)
        self.roi_update_interval = timedelta(seconds=self.config.roi_update_interval)
        self.framegrab_interval = timedelta(seconds=self.config.framegrab_interval)
        self.behavior_check_interval = timedelta(seconds=self.config.behavior_check_interval)
        self.video_split_interval = timedelta(hours=self.config.video_split_hours)
        self.picamera_kwargs = {'framerate': self.config.framerate,
                                'resolution': (self.config.h_resolution, self.config.v_resolution)}
        self.collector = DataCollector(self.video_dir, self.picamera_kwargs)
        self.roi_detector = DetectorBase(MODEL_DIR / self.config.roi_model, self.config.roi_confidence_thresh)
        self.ooi_detector = DetectorBase(MODEL_DIR / self.config.ooi_model, self.config.ooi_confidence_thresh)
        self.behavior_recognizer = BehaviorRecognizer()
        self.notifier = Notifier(self.config.user_email, self.config.sendgrid_from_email, self.config.sendgrid_api_key)
        self.uploader = Uploader(self.project_dir, self.config.cloud_data_dir, self.config.framerate)
        logger.debug('runner initiated')

    def run(self):
        current_datetime = datetime.now()
        if self.start_time < current_datetime.time() < self.end_time:
            logger.info('entering active collection mode')
            self.active_mode()
        else:
            self.passive_mode()

    def active_mode(self):
        self.collector.start_recording()
        current_datetime = datetime.now()
        next_video_split = (current_datetime + self.video_split_interval).replace(minute=0, second=0, microsecond=0)
        next_roi_update = current_datetime
        next_behavior_check = current_datetime + self.behavior_check_interval
        roi_det, roi_slice = None, None

        while self.start_time < current_datetime.time() < self.end_time:
            next_framegrab = current_datetime + self.framegrab_interval
            img = self.collector.capture_frame()
            if current_datetime >= next_roi_update:
                roi_det = self.roi_detector.detect(img)
                if roi_det:
                    roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax,
                                roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
                    next_roi_update = current_datetime + self.roi_update_interval
            if roi_slice:
                img = img[roi_slice]
                occupancy = len(self.ooi_detector.detect(img))
                thumbnail = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10))
                self.behavior_recognizer.append_data(occupancy, thumbnail)
            if current_datetime >= next_behavior_check:
                if self.behavior_recognizer.check_for_behavior():
                    # TODO: fill in placeholder attachment path and message
                    mp4_path = self.video_dir / f'eventclip_{int(current_datetime.timestamp())}'
                    self.behavior_recognizer.thumbnails_to_mp4(mp4_path)
                    notification = Notification(subject=f'possible behavioral event in {self.config.project_id}',
                                                message='',
                                                attachment_path=str(mp4_path))
                    self.notifier.send_email(notification)
            if current_datetime >= next_video_split:
                self.collector.split_recording()
            pause.until(next_framegrab)
            current_datetime = datetime.now()
        self.collector.stop_recording()
        self.notifier.reset()

    def passive_mode(self):
        logger.info('entering passive upload mode')
        self.uploader.convert_and_upload()
        current_datetime = datetime.now()
        next_start = current_datetime.replace(hour=self.config.start_hour, minute=0, second=0, microsecond=0)
        if current_datetime.time() > self.end_time:
            next_start = next_start + timedelta(days=1)
        logger.info(f'pausing until {next_start}')
        pause.until(next_start)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_id', '--pid',
                        type=str,
                        help='Unique project id. If a project with that ID exists, data collection will begin.'
                             'Otherwise, a new project with that ID will be created and the program will exit so that'
                             'you can edit the default config.yaml file if necessary.',
                        default=None)

    # parser.add_argument('--test',
    #                     action='store_true',
    #                     help='run a suite of automated tests to diagnose potential problems')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    config_path = DEFAULT_DATA_DIR / opt.project_id / 'config.yaml'
    if config_path.exists():
        runner = Runner(config_path)
        runner.run()
    else:
        new_project(config_path)

