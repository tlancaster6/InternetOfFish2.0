import pathlib
import sys
import argparse
from datetime import datetime, timedelta, time
import numpy as np
import pause
import logging
import cv2
from logging.handlers import RotatingFileHandler

from modules.data_collection import DataCollector, MockDataCollector
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
TESTING_RESOURCEC_DIR = REPO_ROOT_DIR / 'resources'
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))
if not LOG_DIR.exists():
    LOG_DIR.mkdir()

# initiate logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(name)-16s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = RotatingFileHandler(str(LOG_DIR / 'debug.log'), maxBytes=500000, backupCount=2)
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
    print(f'new project config generated and saved to {config_path}. Edit this file if desired'
          'then re-run main.py to initiate data collection. To enable email notifications, '
          'provide valid entries for "sendgrid_api_key", "sendgrid_from_email", and "user_email". To enable automated '
          'uploads, ensure you have properly installed and configured rclone and provide a valid entry for the '
          '"cloud_data_dir" in the form "rclone_remote:/full/path/to/upload/directory. ')


class Runner:

    def __init__(self, config_path: pathlib.Path):
        logger.debug('beginning runner initialization')
        self.project_dir = config_path.parent
        self.video_dir = self.project_dir / 'Videos'
        self.config = ConfigManager(config_path).config_as_namespace()

        self.start_time = time(hour=self.config.start_hour)
        self.end_time = time(hour=self.config.end_hour)
        logger.debug(f'data collection will run from {self.start_time} to {self.end_time} each day')
        self.roi_update_interval = timedelta(seconds=self.config.roi_update_interval)
        logger.debug(f'ROI update interval set to {self.roi_update_interval}')
        self.framegrab_interval = timedelta(seconds=self.config.framegrab_interval)
        logger.debug(f'Framegrab interval set to {self.framegrab_interval}')
        self.behavior_check_interval = timedelta(seconds=self.config.behavior_check_interval)
        logger.debug(f'Behavior check interval set to {self.behavior_check_interval}')
        self.video_split_interval = timedelta(hours=self.config.video_split_hours)
        logger.debug(f'Video split interval set to {self.video_split_interval}')
        self.picamera_kwargs = {'framerate': self.config.framerate,
                                'resolution': (self.config.h_resolution, self.config.v_resolution)}
        logger.debug(f'camera framerate set to: {self.picamera_kwargs["framerate"]}')
        logger.debug(f'camera resolution set to: {self.picamera_kwargs["resolution"]}')

        self.roi_detector = DetectorBase(MODEL_DIR / self.config.roi_model, self.config.roi_confidence_thresh)
        self.ooi_detector = DetectorBase(MODEL_DIR / self.config.ooi_model, self.config.ooi_confidence_thresh)
        self.behavior_recognizer = BehaviorRecognizer(self.config)
        self.notifier = Notifier(self.config.user_email, self.config.sendgrid_from_email, self.config.sendgrid_api_key)
        self.uploader = Uploader(self.project_dir, self.config.cloud_data_dir, self.config.framerate)
        if self.config.test:
            self.collector = MockDataCollector(TESTING_RESOURCEC_DIR / 'sample_clip.mp4',
                                               self.config.framegrab_interval)
            logger.info('runner successfully initialized in test mode')
        else:
            self.collector = DataCollector(self.video_dir, self.picamera_kwargs)
            logger.info('runner successfully initialized')

    def run(self):
        logger.info('Entering main run loop. Press Ctrl-C at any time to exit')
        try:
            while True:
                if self.config.test:
                    self.test_mode()
                    return
                current_datetime = datetime.now()
                if self.start_time < current_datetime.time() < self.end_time:
                    self.active_mode()
                else:
                    self.passive_mode()
        except KeyboardInterrupt:
            logger.info('Keyboard Interrupt Detected. Running Cleanup operations, please wait until the program exits')
            self.collector.shutdown()
            logger.info('uploading remaining data, please wait')
            self.uploader.convert_and_upload()
            logger.info('Shutdown complete. Exiting')
            sys.exit(0)

    def test_mode(self):
        logger.info('commencing test')
        first_img = self.collector.capture_frame()
        logger.info('locating ROI')
        roi_det = self.roi_detector.detect(first_img)
        roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax,
                    roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
        logger.info(f'ROI located. ROI slice set to {roi_slice}')
        logger.info(f'Commencing OOI detection')
        mock_timestamp = datetime(year=2000, month=1, day=1, hour=12)
        iter_count = 0
        while True:
            img = self.collector.capture_frame()
            if img is False:
                break
            img = img[roi_slice]
            dets = self.ooi_detector.detect(img)
            occupancy = len(dets)
            logger.info(f'\t{iter_count}| frame {self.collector.current_frame} occupancy: {occupancy}')
            thumbnail = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
            self.behavior_recognizer.append_data(mock_timestamp.timestamp(), occupancy, thumbnail)
            mock_timestamp = mock_timestamp + self.framegrab_interval
            iter_count += 1
        logger.info(
            f'fish detection complete. running behavior recognition with {len(self.behavior_recognizer.data_buffer)} unique occupancy values')
        activity_fraction = self.behavior_recognizer.calc_activity_fraction()
        logger.info(f'double occupancy fraction: {activity_fraction}')
        if self.behavior_recognizer.check_for_behavior():
            logger.info('behavior event recognized. Preparing clip.')
            self.video_dir.mkdir(exist_ok=True, parents=True)
            mp4_path = self.video_dir / f'eventclip_{int(datetime.now().timestamp())}.mp4'
            self.behavior_recognizer.thumbnails_to_mp4(mp4_path)
            logger.info('sending email notification')
            notification = Notification(subject=f'possible behavioral event in {self.config.project_id}',
                                        message=f'activity fraction: {self.behavior_recognizer.calc_activity_fraction()}',
                                        attachment_path=str(mp4_path))
            self.notifier.notify(notification)
        else:
            logger.info('behavioral event not recognized')
        logger.info('uploading results')
        self.uploader.convert_and_upload()
        logger.info('test complete. exiting.')


    def active_mode(self):
        logger.info('entering active collection mode')
        self.collector.start_recording()
        current_datetime = datetime.now()
        next_video_split = (current_datetime + self.video_split_interval).replace(minute=0, second=0, microsecond=0)
        next_roi_update = current_datetime
        next_behavior_check = current_datetime + self.behavior_check_interval
        roi_det, roi_slice = None, None

        while self.start_time < current_datetime.time() < self.end_time:
            next_framegrab = current_datetime + self.framegrab_interval
            img = self.collector.capture_frame()
            if img.size == 0:
                logger.debug(f'invalid image with shape {img.shape} encountered. Skipping')
                pause.until(next_framegrab)
                current_datetime = datetime.now()
            if current_datetime >= next_roi_update:
                roi_det = self.roi_detector.detect(img)
                if roi_det:
                    roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax,
                                roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
                    next_roi_update = current_datetime + self.roi_update_interval
            if roi_slice:
                img = img[roi_slice]
                dets = self.ooi_detector.detect(img)
                occupancy = len(self.ooi_detector.detect(img))

                for det in dets:
                    bbox = det.bbox
                    cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)

                thumbnail = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
                thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
                self.behavior_recognizer.append_data(current_datetime.timestamp(), occupancy, thumbnail)
            if current_datetime >= next_behavior_check:
                if self.behavior_recognizer.check_for_behavior():
                    logger.info('possible behavioral event. Sending notification')
                    mp4_path = self.video_dir / f'eventclip_{int(current_datetime.timestamp())}.mp4'
                    self.behavior_recognizer.thumbnails_to_mp4(mp4_path)
                    notification = Notification(subject=f'possible behavioral event in {self.config.project_id}',
                                                message=f'activity fraction: {self.behavior_recognizer.calc_activity_fraction()}',
                                                attachment_path=str(mp4_path))
                    self.notifier.send_email(notification)
                    next_behavior_check = next_behavior_check + self.behavior_check_interval
            if current_datetime >= next_video_split:
                self.collector.split_recording()
                next_video_split = next_video_split + self.video_split_interval
                # if the video is going to split less than an hour before the end time, prevent it
                if next_video_split.hour == self.end_time.hour:
                    next_video_split = next_video_split + timedelta(hours=1)
            pause.until(next_framegrab)
            current_datetime = datetime.now()
        self.collector.stop_recording()
        self.notifier.reset()

    def passive_mode(self):
        logger.info('entering passive upload mode')
        logger.info('converting and uploading videos')
        self.uploader.convert_and_upload()
        logger.info('conversion and upload complete')
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

    parser.add_argument('--test',
                        action='store_true',
                        help='run a suite of automated tests to diagnose potential problems')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    if opt.test:
        config_path = DEFAULT_DATA_DIR / 'test_project' / 'config.yaml'
        if not config_path.exists():
            config_manager = ConfigManager(config_path)
            config_manager.generate_test_config()
            print(
                'default config generated. To test email notification and automated uploads, open the config.yaml file now '
                '(located under projects/interactive_test), provide appropriate values for cloud_data_dir, user_email, '
                'sendgrid_api_key, and sendgrid_from_email, then save and close the config file. Or leave the config.yaml '
                'file alone to skip these tests. ')
            input('Press enter when ready to resume testing')
        runner = Runner(config_path)
        runner.run()

    else:
        config_path = DEFAULT_DATA_DIR / opt.project_id / 'config.yaml'
        if config_path.exists():
            runner = Runner(config_path)
            runner.run()
        else:
            new_project(config_path)
