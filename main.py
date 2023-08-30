import argparse
from datetime import datetime, timedelta, time
from time import sleep
import pathlib
import sys
import os
import numpy as np
from modules.data_collection import DataCollector
from modules.object_detection import DetectorBase
from modules.upload_automation import Uploader
from modules.behavior_recognition import BehaviorRecognizer
from modules.config_manager import ConfigManager
import pause

# establish a stable location within the filesystem
FILE = pathlib.Path(__file__).resolve()
REPO_ROOT_DIR = FILE.parents[0]  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
# ROOT = pathlib.Path(os.path.relpath(REPO_ROOT_DIR, pathlib.Path.cwd()))  # relative to working directory
MODEL_DIR = REPO_ROOT_DIR / 'models'
DATA_DIR = REPO_ROOT_DIR / 'projects'
SENDGRID_CREDENTIAL_FILE = REPO_ROOT_DIR / 'credentials' / 'sendgrid_key.secret'

POLLING_INTERVAL = timedelta(seconds=0.01)


def main(project_id):
    project_dir = DATA_DIR / project_id
    config_path = project_dir / 'config.yaml'
    config_manager = ConfigManager(config_path)
    if not config_path.exists():
        project_dir.mkdir(parents=True)
        config_manager.generate_new_config(project_id)
        print('new project config generated. Edit this file if desired, then re-run main.py to \n'
              'initiate data collection. Note that, to enable email notifications and rclone uploads, \n'
              'you must supply the "cloud_data_dir" and "user_email" fields manually in the config \n')

        return
    config = config_manager.config_as_namespace()
    video_dir = project_dir / 'Videos'

    start_time = time(hour=config.start_hour)
    end_time = time(hour=config.end_hour)

    roi_update_interval = timedelta(seconds=config.roi_update_interval)
    framegrab_interval = timedelta(seconds=config.framegrab_interval)
    video_split_interval = timedelta(hours=config.video_split_hours)

    picamera_kwargs = {'framerate': config.framerate, 'resolution': (config.h_resolution, config.v_resolution)}

    while True:
        current_datetime = datetime.now()
        if start_time < current_datetime.time() < end_time:
            collector = DataCollector(video_dir, picamera_kwargs)
            roi_detector = DetectorBase(MODEL_DIR / config.roi_model, config.roi_confidence_thresh)
            ooi_detector = DetectorBase(MODEL_DIR / config.ooi_model, config.ooi_confidence_thresh)
            behavior_recognizer = BehaviorRecognizer()
            collector.start_recording()

            next_video_split = (current_datetime + video_split_interval).replace(minute=0, second=0, microsecond=0)
            next_framegrab = current_datetime + framegrab_interval
            next_roi_update = current_datetime + roi_update_interval

            roi_det, roi_slice = None, None

            while start_time < current_datetime.time() < end_time:
                next_poll = current_datetime + POLLING_INTERVAL
                if current_datetime >= next_video_split:
                    collector.split_recording()
                if current_datetime >= next_framegrab:
                    img = collector.capture_frame()
                    if current_datetime >= next_roi_update:
                        roi_det = roi_detector.detect(img)
                        if roi_det:
                            roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax,
                                              roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
                    if roi_slice:
                        img = img[roi_slice]
                        ooi_dets = ooi_detector.detect(img)
                        behavior_recognizer.append_dets(ooi_dets)
                pause.until(next_poll)
                current_datetime = datetime.now()
            collector.stop_recording()
            roi_detector, ooi_detector, behavior_recognizer, collector = None, None, None, None
        else:
            uploader = Uploader(project_dir, config.cloud_data_dir, config.framerate)
            uploader.convert_and_upload()
            current_datetime = datetime.now()
            next_start = current_datetime.replace(hour=config.start_hour, minute=0, second=0, microsecond=0)
            if current_datetime.time() > end_time:
                next_start = next_start + timedelta(days=1)
            pause.until(next_start)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_id',
                        type=str,
                        help='unique name for this project.',
                        default='default_project')

    # parser.add_argument('--test',
    #                     action='store_true',
    #                     help='run a suite of automated tests to diagnose potential problems')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt.project_id)
