from time import perf_counter
import pathlib
import sys
import cv2
import pandas as pd
from statistics import mean, median, stdev

FILE = pathlib.Path(__file__).resolve()
TESTING_DIR = FILE.parent
REPO_ROOT_DIR = TESTING_DIR.parent  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
TESTING_RESOURCE_DIR = TESTING_DIR / 'resources'

from modules.object_detection import DetectorBase


class BenchMarker:

    def __init__(self):
        self.roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')
        self.ooid = DetectorBase(TESTING_RESOURCE_DIR / 'ooi.tflite')
        self.img_full, self.img_croppped = self.load_testing_images()
        self.log_path = REPO_ROOT_DIR / 'logs' / 'benchmarking.log'

    def run_benchmarks(self):

        roi_det_metrics = self.time_roi_detection()
        ooi_det_metrics = self.time_ooi_detection()
        df = pd.DataFrame.from_records([roi_det_metrics, ooi_det_metrics])
        df.to_csv(str(self.log_path))

    def load_testing_images(self):
        img_full = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
        img_cropped = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [img_full, img_cropped]]

    def _detection_benchmark(self, identifier, detector_object: DetectorBase, img, reps):
        print(f'timing detection for {reps} reps')
        times = []
        for _ in range(reps):
            start = perf_counter()
            dets = detector_object.detect(img)
            end = perf_counter()
            times.append(end - start)
        metrics = {
            'identifier': identifier,
            'mean': mean(times),
            'median': median(times),
            'min': min(times),
            'max': max(times),
            'stdev': stdev(times),
            'n': reps
        }
        return metrics

    def time_roi_detection(self, reps=500):
        return self._detection_benchmark('roi_det_time', self.roid, self.img_full)

    def time_ooi_detection(self, reps=500):
        return self._detection_benchmark('ooi_det_time', self.ooid, self.img_croppped)