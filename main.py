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

# establish a stable location within the filesystem
FILE = pathlib.Path(__file__).resolve()
LOCAL_ROOT = FILE.parents[0]  # repository root
if str(LOCAL_ROOT) not in sys.path:
    sys.path.append(str(LOCAL_ROOT))  # add LOCAL_ROOT to system path
ROOT = pathlib.Path(os.path.relpath(LOCAL_ROOT, pathlib.Path.cwd()))  # relative to working directory
MODEL_DIR = ROOT / 'models'

POLLING_INTERVAL = 0.01


def main(opt):
    project_dir = ROOT / opt.project_id
    video_dir = project_dir / 'Videos'
    cloud_root = pathlib.PurePath(opt.cloud_root)

    start_time = time(hour=opt.start_hour)
    end_time = time(hour=opt.end_hour)

    roi_update_interval = timedelta(seconds=opt.roi_update_interval)
    framegrab_interval = timedelta(seconds=opt.framegrab_interval)
    video_split_interval = timedelta(hours=opt.video_split_hours)

    picamera_kwargs = {'framerate': opt.framerate, 'resolution': (opt.h_resolution, opt.v_resolution)}

    while True:
        current_time = datetime.now().time()
        if start_time < current_time < end_time:
            collector = DataCollector(video_dir, picamera_kwargs)
            roi_detector = DetectorBase(MODEL_DIR / opt.roi_model, opt.roi_confidence_thresh)
            ooi_detector = DetectorBase(MODEL_DIR / opt.ooi_model, opt.ooi_confidence_thresh)
            behavior_recognizer = BehaviorRecognizer()
            collector.start_recording()

            next_video_split = (datetime.now() + video_split_interval).time().replace(minute=0, second=0, microsecond=0)
            next_framegrab = (datetime.now() + framegrab_interval).time()
            next_roi_update = (datetime.now() + roi_update_interval).time()

            roi_det, roi_slice = None, None

            while start_time < current_time < end_time:
                current_time = datetime.now().time()
                if current_time >= next_video_split:
                    collector.split_recording()
                if current_time >= next_framegrab:
                    img = collector.capture_frame()
                    if current_time >= next_roi_update:
                        roi_det = roi_detector.detect(img)
                        if roi_det:
                            roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax,
                                              roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
                    if roi_slice:
                        img = img[roi_slice]
                        ooi_dets = ooi_detector.detect(img)
                        behavior_recognizer.append_dets(ooi_dets)
                sleep(POLLING_INTERVAL)
            collector.stop_recording()
            roi_detector, ooi_detector, behavior_recognizer, collector = None, None, None, None
        else:
            uploader = Uploader(ROOT, cloud_root, opt.project_id)
            uploader.upload_all()
            while current_time > end_time or current_time < start_time:
                current_time = datetime.now().time()
                sleep(POLLING_INTERVAL)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_id',
                        type=str,
                        help='unique name for this project.',
                        default='default_project')

    parser.add_argument('--cloud_root',
                        type=str,
                        help='Destination for cloud uploads. If None (default) automatic uploads will be skipped.',
                        default=None)

    parser.add_argument('--roi_model',
                        type=str,
                        help='full name of the tflite file for the region-of-interest detection model.',
                        default='pipe_detector.tflite')

    parser.add_argument('--ooi_model',
                        type=str,
                        help='full name of the tflite file for the object-of-interest (ooi) detection model.',
                        default='fish_detector.tflite'
                        )

    parser.add_argument('--roi_confidence_thresh',
                        type=float,
                        help='minimum confidence for a region-of-interest detection to be considered valid.',
                        default=0.75)

    parser.add_argument('--ooi_confidence_thresh',
                        type=float,
                        help='minimum confidence for an object-of-interest detection to be considered valid.',
                        default=0.25)

    parser.add_argument('--framerate',
                        type=int,
                        help='picamera framerate in fps',
                        default=30)

    parser.add_argument('--h_resolution',
                        type=int,
                        help='picamera horizontal resolution',
                        default=1280)

    parser.add_argument('--v_resolution',
                        type=int,
                        help='picamera vertical resolution',
                        default=976)

    parser.add_argument('--framegrab_interval',
                        type=float,
                        help='seconds between successive frame-grabs.',
                        default=0.2)

    parser.add_argument('--roi_update_interval',
                        type=float,
                        help='seconds between successive ROI updates. Must be greater than or equal to '
                             'framegrab_interval. Use small numbers if the ROI can move',
                        default=600)

    parser.add_argument('--start_hour',
                        type=int,
                        help='when to start collecting data. Defaults to 7, meaning data collection starts at 7:00am '
                             'each day.',
                        default=7)

    parser.add_argument('--end_hour',
                        type=int,
                        help='when to stop collecting data. Defaults to 19, meaning data collection ends at 19:00, '
                             'or 7:00pm, each day.',
                        default=19)

    parser.add_argument('--video_split_hours',
                        type=int,
                        help='maximum length, in hours, of each recorded video. Useful for keeping individual file'
                             'sizes manageable. The switch from one file destination to the next is usually seamless,'
                             'with at most a few dropped frames',
                        default=3
                        )

    parser.add_argument()
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
