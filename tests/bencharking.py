from time import perf_counter
import pathlib
import sys
import cv2
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
        with open(str(self.log_path), 'w') as f:
            f.write('roi detection benchmarking\n')
            for k, v in roi_det_metrics.items():
                f.write(f'\t{k}: {v})')
            f.write('ooi detection benchmarking\n')
            for k, v in ooi_det_metrics.items():
                f.write(f'\t{k}: {v})')

    def load_testing_images(self):
        img_full = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
        img_cropped = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [img_full, img_cropped]]

    def _detection_benchmark(self, detector_object: DetectorBase, img, reps):
        print(f'timing detection for {reps} reps')
        times = []
        for _ in range(reps):
            start = perf_counter()
            dets = detector_object.detect(img)
            end = perf_counter()
            times.append(end - start)
        metrics = {
            'mean': mean(times),
            'median': median(times),
            'min': min(times),
            'max': max(times),
            'stdev': stdev(times),
            'n': reps
        }
        return metrics

    def time_roi_detection(self, reps=500):
        return self._detection_benchmark(self.roid, self.img_full, reps=reps)

    def time_ooi_detection(self, reps=500):
        return self._detection_benchmark(self.ooid, self.img_croppped, reps=reps)