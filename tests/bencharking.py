from time import perf_counter
import pathlib
import sys
import cv2
from statistics import mean, median, stdev
import json
import pandas as pd

FILE = pathlib.Path(__file__).resolve()
TESTING_DIR = FILE.parent
REPO_ROOT_DIR = TESTING_DIR.parent  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
TESTING_RESOURCE_DIR = TESTING_DIR / 'resources'

from modules.object_detection import DetectorBase


class SpeedBenchMarker:

    def __init__(self):
        self.roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')
        self.ooid = DetectorBase(TESTING_RESOURCE_DIR / 'ooi.tflite')
        self.img_full, self.img_croppped = self.load_testing_images()
        self.log_path = REPO_ROOT_DIR / 'logs' / 'benchmark_results.json'

    def run_benchmarks(self, reps=500):
        metrics = {}
        metrics.update({'roi_det_metrics': self.time_roi_detection(reps=reps)})
        metrics.update({'ooi_det_metrics': self.time_ooi_detection(reps=reps)})
        with open(str(self.log_path), 'w') as f:
            json.dump(metrics, f, indent=4)


    def load_testing_images(self):
        img_full = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
        img_cropped = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [img_full, img_cropped]]

    def _detection_benchmark(self, detector_object: DetectorBase, img, reps):
        print(f'timing detection for {reps} reps')
        records = []
        for _ in range(reps):
            dets, times = detector_object._timed_detect(img)
            records.append(times)

        metrics = {}
        for stage in ['preprocessing', 'inference', 'postprocessing']:
            times = [rec[stage] for rec in records]
            summary = {
                'mean': mean(times),
                'median': median(times),
                'min': min(times),
                'max': max(times),
                'stdev': stdev(times),
                'n': reps
            }
            metrics.update({stage: summary})
        return metrics

    def time_roi_detection(self, reps=500):
        return self._detection_benchmark(self.roid, self.img_full, reps=reps)

    def time_ooi_detection(self, reps=500):
        return self._detection_benchmark(self.ooid, self.img_croppped, reps=reps)


class AccuracyBenchMarker:

    def __init__(self, video_path):
        self.roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite', confidence_thresh=0.1)
        self.video_path = pathlib.Path(video_path)
        self.output_path = self.video_path.with_suffix('.csv')

    def run_detection(self):
        cap = cv2.VideoCapture(str(self.video_path))
        ret = True
        current_frame = 0
        rows = []
        columns = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'score']
        while ret:
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = self.roid.detect(img)
            for det in dets:
                rows.append([current_frame, det.bbox.xmin, det.bbox.xmax, det.bbox.ymin, det.bbox.ymax, det.score])
            current_frame += 1
        cap.release()
        df = pd.DataFrame(rows, columns=columns).set_index('frame')
        df.to_csv(str(self.output_path))





