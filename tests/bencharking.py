from time import perf_counter
import pathlib
import sys
import cv2

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
        self.log_path = REPO_ROOT_DIR / 'benchmarking.log'

    def run_benchmarks(self):
        metrics = {}
        metrics.update({'roi_det_time': self.time_roi_detection()})
        metrics.update({'ooi_det_time': self.time_ooi_detection()})
        with open(str(self.log_path), 'w') as f:
            for key, val in metrics.items():
                f.write(f'{key}: val')

    def load_testing_images(self):
        img_full = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
        img_cropped = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [img_full, img_cropped]]

    def time_roi_detection(self, reps=50):
        print(f'timing region-of-interest detection for {reps} reps')
        start = perf_counter()
        for _ in range(reps):
            dets = self.roid.detect(self.img_full)
        end = perf_counter()
        det_time = (end - start) / reps
        return det_time

    def time_ooi_detection(self, reps=50):
        print(f'timing object-of-interest detection for {reps} reps')
        start = perf_counter()
        for _ in range(reps):
            dets = self.ooid.detect(self.img_croppped)
        end = perf_counter()
        det_time = (end - start) / reps
        return det_time
