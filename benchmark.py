import pathlib
import sys
import cv2
import os
import shutil
from statistics import mean, median, stdev
import pandas as pd
from modules.object_detection import DetectorBase
import numpy as np
# establish filesystem locations
FILE = pathlib.Path(__file__).resolve()
REPO_ROOT_DIR = FILE.parent  # repository root
MODEL_DIR = REPO_ROOT_DIR / 'models'
RESOURCE_DIR = REPO_ROOT_DIR / 'resources'
DEFAULT_DATA_DIR = REPO_ROOT_DIR / 'projects'
LOG_DIR = REPO_ROOT_DIR / 'logs'
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))
if not LOG_DIR.exists():
    LOG_DIR.mkdir()


class BenchMarker:

    def __init__(self):
        pass

    def benchmark_detection_speed(self, detector_object, img,  reps=500):
        print(f'timing detection for {reps} reps')
        records = []
        for _ in range(reps):
            dets, times = detector_object._timed_detect(img)
            records.append(times)

        metrics = []
        for stage in ['preprocessing', 'inference', 'postprocessing']:
            times = [rec[stage] for rec in records]
            metrics.extend([
                [stage, 'mean', mean(times)],
                [stage, 'median', median(times)],
                [stage, 'min', min(times)],
                [stage, 'max', max(times)],
                [stage, 'stdev', stdev(times)],
                [stage, 'n', reps]
            ])
        metrics = pd.DataFrame(metrics, columns=['stage', 'metric', 'value'])
        return metrics

    def benchmark_roid_speed(self):
        roid = DetectorBase(MODEL_DIR / 'roi.tflite')
        img = cv2.cvtColor(cv2.imread(str(RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        metrics = self.benchmark_detection_speed(roid, img)
        metrics.to_csv(str(LOG_DIR / 'roi_speed_benchmark.log'))

    def benchmark_ooid_speed(self):
        ooid = DetectorBase(MODEL_DIR / 'ooi.tflite')
        img = cv2.cvtColor(cv2.imread(str(RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        metrics = self.benchmark_detection_speed(ooid, img)
        metrics.to_csv(str(LOG_DIR / 'ooi_speed_benchmark.log'))

    def realistic_benchmark(self):
        # set up the test
        tmp_dir = pathlib.Path.home() / 'tmp'
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(RESOURCE_DIR / 'sample_clip.mp4'))
        frame_counter = 0
        img_paths = []
        while True:
            ret, frame = cap.read()
            if ret:
                img_paths.append(tmp_dir / f'frame{frame_counter}.jpg')
                cv2.imwrite(str(img_paths[-1]), frame)
                frame_counter += 1
            else:
                break
        roid = DetectorBase(MODEL_DIR / 'roi.tflite')
        ooid = DetectorBase(MODEL_DIR / 'ooi.tflite')

        # run roi detection
        img = cv2.cvtColor(cv2.imread(str(img_paths[0]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roi_det, roid_times = roid._timed_detect(img)
        roi_slice = np.s_[roi_det[0].bbox.ymin:roi_det[0].bbox.ymax, roi_det[0].bbox.xmin:roi_det[0].bbox.xmax]
        roid_metrics = pd.Series(roid_times)
        # run ooi detection
        records = []
        for img_p in img_paths:
            img = cv2.cvtColor(cv2.imread(str(img_p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)[roi_slice]
            dets, times = ooid._timed_detect(img)
            records.append(times)

        ooid_metrics = []
        for stage in ['preprocessing', 'inference', 'postprocessing']:
            times = [rec[stage] for rec in records]
            ooid_metrics.extend([
                [stage, 'mean', mean(times)],
                [stage, 'median', median(times)],
                [stage, 'min', min(times)],
                [stage, 'max', max(times)],
                [stage, 'stdev', stdev(times)],
            ])
        ooid_metrics = pd.DataFrame(ooid_metrics, columns=['stage', 'metric', 'value'])
        ooid_metrics.to_csv(str(LOG_DIR / 'ooi_speed_benchmark2.log'))
        roid_metrics.to_csv(str(LOG_DIR / 'roi_speed_benchmark2.log'))

        # clean up
        shutil.rmtree(tmp_dir)
