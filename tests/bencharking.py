from time import perf_counter
import pathlib
import sys
import cv2
from statistics import mean, median, stdev
import json
import pandas as pd
import numpy as np
from types import SimpleNamespace
from itertools import product
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

FILE = pathlib.Path(__file__).resolve()
TESTING_DIR = FILE.parent
REPO_ROOT_DIR = TESTING_DIR.parent  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
TESTING_RESOURCE_DIR = TESTING_DIR / 'resources'

# from modules.object_detection import DetectorBase
from modules.behavior_recognition import BehaviorRecognizer


# def generate_dense_detection_data(video_path: pathlib.Path):
#     print(f'processing {video_path.name}')
#     roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite', confidence_thresh=0.1)
#     cap = cv2.VideoCapture(str(video_path))
#     current_frame = 0
#     rows = []
#     columns = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'score']
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             break
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         dets = roid.detect(img)
#         for det in dets:
#             rows.append([current_frame, det.bbox.xmin, det.bbox.xmax, det.bbox.ymin, det.bbox.ymax, det.score])
#         current_frame += 1
#     cap.release()
#     df = pd.DataFrame(rows, columns=columns).set_index('frame')
#     df.to_csv(str(video_path.with_suffix('.csv')))
#
#
# class SpeedBenchMarker:
#
#     def __init__(self):
#         self.roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')
#         self.ooid = DetectorBase(TESTING_RESOURCE_DIR / 'ooi.tflite')
#         self.img_full, self.img_croppped = self.load_testing_images()
#         self.log_path = REPO_ROOT_DIR / 'logs' / 'benchmark_results.json'
#
#     def run_benchmarks(self, reps=500):
#         metrics = {}
#         metrics.update({'roi_det_metrics': self.time_roi_detection(reps=reps)})
#         metrics.update({'ooi_det_metrics': self.time_ooi_detection(reps=reps)})
#         with open(str(self.log_path), 'w') as f:
#             json.dump(metrics, f, indent=4)
#
#
#     def load_testing_images(self):
#         img_full = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
#         img_cropped = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
#         return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [img_full, img_cropped]]
#
#     def _detection_benchmark(self, detector_object: DetectorBase, img, reps):
#         print(f'timing detection for {reps} reps')
#         records = []
#         for _ in range(reps):
#             dets, times = detector_object._timed_detect(img)
#             records.append(times)
#
#         metrics = {}
#         for stage in ['preprocessing', 'inference', 'postprocessing']:
#             times = [rec[stage] for rec in records]
#             summary = {
#                 'mean': mean(times),
#                 'median': median(times),
#                 'min': min(times),
#                 'max': max(times),
#                 'stdev': stdev(times),
#                 'n': reps
#             }
#             metrics.update({stage: summary})
#         return metrics
#
#     def time_roi_detection(self, reps=500):
#         return self._detection_benchmark(self.roid, self.img_full, reps=reps)
#
#     def time_ooi_detection(self, reps=500):
#         return self._detection_benchmark(self.ooid, self.img_croppped, reps=reps)

class ParameterSweeper:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_paths = pathlib.Path(data_dir).glob('*.csv')
        self.data = self.read_data()

    def read_data(self):
        return {dp.stem: pd.read_csv(str(dp), index_col=0) for dp in self.data_paths}

    def convert_dets_to_occupancy(self, clip_name: str, confidence_thresh: float):
        occupancy = self.data[clip_name]
        occupancy = (occupancy['score'] >= confidence_thresh).groupby(level=0).sum()
        occupancy.name = 'occupancy'
        occupancy = occupancy.reindex(range(occupancy.index.max() + 1), fill_value=0)
        return pd.DataFrame(occupancy)

    def clip_name_to_label(self, clip_name):
        return 0 if 'nonspawning' in clip_name else 1

    def calc_param_grid_size(self, param_grid):
        total_combos = 1
        for grid_edge in param_grid.values():
            total_combos *= len(grid_edge)
        return total_combos

    def generate_sweep_data(self, param_grid, output_path):
        column_names = list(param_grid.keys()) + list(self.data.keys())
        rows = []
        param_grid_size = self.calc_param_grid_size(param_grid)
        print(f'generating sweep data for {param_grid_size} unique parameter sets')
        for param_set in tqdm(product(*list(param_grid.values())), total=param_grid_size):
            config = SimpleNamespace(**{k: v for k, v in list(zip(param_grid.keys(), param_set))})
            config.behavior_check_window = 60
            br = BehaviorRecognizer(config)
            framegrab_step = int(config.framegrab_interval * 30)
            activity_fractions = []
            for clip_name, clip_data in self.data.items():
                clip_data = self.convert_dets_to_occupancy(clip_name, config.ooi_confidence_thresh)
                clip_data['timestamp'] = clip_data.index / 30
                clip_data = clip_data[clip_data.index % framegrab_step == 0]
                for row in clip_data.itertuples():
                    br.append_data(row.timestamp, row.occupancy, None)
                activity_fractions.append(br.calc_activity_fraction())
                br.reset()
            if sum(activity_fractions):
                rows.append(list(param_set) + activity_fractions)
        df = pd.DataFrame(rows, columns=column_names)

        print(f'sweep data generated\n'
              f'{param_grid_size} unique parameter sets tested\n'
              f'{len(df)} of param sets yielded non-trivial results\n'
              )
        print(f'saving non-trivial results to {output_path}')
        df.to_hdf(str(output_path), key='df')
        print('done')

    def execute_sweep(self, input_path, output_path):
        print(f'executing sweep on {input_path.name}')
        clip_names = list(self.data.keys())
        df = pd.read_hdf(input_path)
        data_df = df[clip_names]
        labels = np.array([self.clip_name_to_label(clip_name) for clip_name in list(self.data.keys())])
        model = LogisticRegressionCV(random_state=0)
        accuracies = []
        boundaries = []
        for idx, row in data_df.iterrows():
            x, y = row.values.reshape(-1, 1), labels
            model.fit(x, y)
            accuracies.append(model.score(x, y))
            boundaries.append(np.nan if model.coef_[0][0] == 0 else -model.intercept_[0]/model.coef_[0][0])
        output_df = df[[cname for cname in df.columns if cname not in clip_names]].copy()
        output_df['accuracy'] = accuracies
        output_df['boundary'] = boundaries
        output_df.to_hdf(output_path, key='df')

    def plot_sweep_results(self, input_path):
        pass


clustering_param_grid = {
    'ooi_confidence_thresh': np.arange(0.1, 0.9, 0.05),
    'framegrab_interval': np.arange(1, 31, 1) / 30,
    'behavior_max_gap_within_event': np.arange(0, 5.5, 0.5),
    'behavior_min_event_length': np.arange(0, 10, 1),
    'behavior_min_individuals_roi': [2],
    'behavior_max_individuals_roi': [2, 3],
    'behavior_use_dbscan': [True]
}

nonclustering_param_grid = {
    'ooi_confidence_thresh': np.arange(0.1, 0.9, 0.05),
    'framegrab_interval': np.arange(1, 31, 1) / 30,
    'behavior_max_gap_within_event': [0],
    'behavior_min_event_length': [0],
    'behavior_min_individuals_roi': [2],
    'behavior_max_individuals_roi': [2, 3],
    'behavior_use_dbscan': [False]
}

test_param_grid = {
    'ooi_confidence_thresh': [0.2, 0.4],
    'framegrab_interval': [5/30, 10/30],
    'behavior_max_gap_within_event': [0.5, 2.0],
    'behavior_min_event_length': [1, 5],
    'behavior_min_individuals_roi': [1, 2],
    'behavior_max_individuals_roi': [2, 3],
    'behavior_use_dbscan': [True, False]
}


base_dir = pathlib.Path(r"C:\Users\tucke\PycharmProjects\InternetOfFish2.0\projects\benchmarking")
data_dir = base_dir / 'detections'
results_dir = base_dir / 'results'

sweeper = ParameterSweeper(data_dir)
#
sweeper.generate_sweep_data(nonclustering_param_grid, results_dir / 'nonclustering_sweep.h5')
sweeper.execute_sweep(results_dir / 'nonclustering_sweep.h5', results_dir / 'nonclustering_results.h5')

sweeper = ParameterSweeper(data_dir)

# sweeper.generate_sweep_data(clustering_param_grid, results_dir / 'clustering_sweep.h5')
sweeper.execute_sweep(results_dir / 'clustering_sweep.h5', results_dir / 'clustering_results.h5')
