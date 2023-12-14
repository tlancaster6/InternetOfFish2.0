import pathlib
import sys
import cv2
from statistics import mean, median, stdev
import pandas as pd
from modules.object_detection import DetectorBase

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
        self.ooid = DetectorBase(MODEL_DIR / 'ooi.tflite')
        # self.roid = DetectorBase(model_dir / 'roi.tflite')
        self.cropped_sample_img = self.load_sample_img()

    def load_sample_img(self):
        cropped_img = cv2.imread(str(RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
        # full_img = cv2.imread(str(RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
        return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    def benchmark_speed(self, reps=500):
        print(f'timing detection for {reps} reps')
        records = []
        for _ in range(reps):
            dets, times = self.ooid._timed_detect(self.cropped_sample_img)
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
        metrics.to_csv(str(LOG_DIR / 'speed_benchmark.log'))
        return metrics
