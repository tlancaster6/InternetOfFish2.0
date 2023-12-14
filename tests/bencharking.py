from time import perf_counter
import pathlib
import sys
import cv2
from statistics import mean, median, stdev
import json
import pandas as pd
import numpy as np
from types import SimpleNamespace
from dbscan1d import DBSCAN1D
from itertools import product
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

FILE = pathlib.Path(__file__).resolve()
TESTING_DIR = FILE.parent
REPO_ROOT_DIR = TESTING_DIR.parent  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
TESTING_RESOURCE_DIR = TESTING_DIR / 'resources'

from modules.object_detection import DetectorBase
from modules.behavior_recognition import BehaviorRecognizer



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

class ParameterSweeper:

    def __init__(self, labeled_data_dir, noise_level=0):
        self.noise_level = noise_level
        self.labeled_data_dir = pathlib.Path(labeled_data_dir)
        self.data_paths = list(labeled_data_dir.glob('**/*.csv'))
        self.clipname_to_label_mapping = self.map_clipnames_to_labels()
        self.clip_names = list(self.clipname_to_label_mapping.keys())
        self.label_to_int_mapping = self.map_labels_to_ints()
        self.data = self.read_data()

    def read_data(self):
        df = []
        for dp in self.data_paths:
            tmp = pd.read_csv(str(dp), index_col=0)
            tmp['clip_name'] = dp.stem
            tmp['label'] = self.clipname_to_label_mapping[dp.stem]
            df.append(tmp)
        return pd.concat(df)

    def inject_noise(self, noise_level, actual_data):
        noisy_data = actual_data.reset_index()
        nonnull_indices = noisy_data.dropna().index.values
        drop_indices = np.random.choice(nonnull_indices, int(len(nonnull_indices) * noise_level))
        noisy_data.loc[drop_indices, 'score'] = 0
        insertion_frames = np.random.randint(0, max(noisy_data.index), int(noise_level * max(noisy_data.index)))

    def map_clipnames_to_labels(self):
        return {dp.stem: dp.parent.name for dp in self.data_paths}

    def map_labels_to_ints(self):
        string_labels = pd.Series([dp.parent.name for dp in self.data_paths]).unique()
        int_labels = np.arange(len(string_labels))
        return {k: v for k, v in list(zip(string_labels, int_labels))}

    def convert_dets_to_occupancy(self, clip_name: str, confidence_thresh: float):
        occupancy = self.data.query('clip_name==@clip_name')
        occupancy = (occupancy['score'] >= confidence_thresh).groupby(level=0).sum()
        occupancy.name = 'occupancy'
        return pd.DataFrame(occupancy)

    def generate_data_summary(self, output_path, frame_interval=1):
        source_data = self.data[self.data.index % frame_interval == 0]
        fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5))

        data = source_data[['score', 'clip_name']].dropna()
        sns.kdeplot(data, x='score', ax=axes[0][0], hue='clip_name', legend=False)
        axes[0][0].set(title='distribution of object detection\n confidences for each video')

        data = (source_data['xmax'] - source_data['xmin']) * (source_data['ymax'] - source_data['ymin'])
        data = pd.DataFrame(data, columns=['area'])
        data['clip_name'] = source_data['clip_name']
        data = data.dropna()
        sns.kdeplot(data, x='area', ax=axes[0][1], hue='clip_name', legend=False)
        axes[0][1].set(title='distribution of bounding box \nareas (pixels^2) for each video')

        data = []
        for cn in self.clip_names:
            label = self.clipname_to_label_mapping[cn]
            occupancy = self.convert_dets_to_occupancy(cn, 0.1)
            occupancy_fraction = occupancy.value_counts(normalize=True)[2] if 2 in occupancy.values else 0.0
            data.append([label, occupancy_fraction])
        data = pd.DataFrame(data, columns=['label', 'double_occupancy_fraction'])
        sns.histplot(data, x='double_occupancy_fraction', hue='label', ax=axes[1][0], bins=20)
        axes[1][0].set(title='distribution of double occupancy fractions\n score>=0.1')

        data = []
        for cn in self.clip_names:
            label = self.clipname_to_label_mapping[cn]
            occupancy = self.convert_dets_to_occupancy(cn, 0.9)
            occupancy_fraction = occupancy.value_counts(normalize=True)[2] if 2 in occupancy.values else 0.0
            data.append([label, occupancy_fraction])
        data = pd.DataFrame(data, columns=['label', 'double_occupancy_fraction'])
        sns.histplot(data, x='double_occupancy_fraction', hue='label', ax=axes[1][1], bins=20)
        axes[1][1].set(title='distribution of double occupancy fractions\n score>=0.9')

        fig.suptitle('Summary of Benchmarking/Tuning Dataset')
        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)

    def predict_spawning_old(self):
        y_actual = []
        y_pred = []
        for cn in self.clip_names:
            y_actual.append(self.label_to_int_mapping[self.clipname_to_label_mapping[cn]])
            occupancies = self.convert_dets_to_occupancy(cn, 0.5)
            occupancies = occupancies[occupancies.index % 8 == 0]
            hit_counter = 0
            spawning_detected = False
            for occ in occupancies.occupancy:
                if occ == 2:
                    hit_counter += 1
                else:
                    hit_counter -= 1
                if hit_counter < 0:
                    hit_counter = 0
                elif hit_counter == 20:
                    spawning_detected = True
                    break
            y_pred.append(1 if spawning_detected else 0)
        f1 = f1_score(y_actual, y_pred)
        cm = confusion_matrix(y_actual, y_pred)
        return f1, cm





    def calc_param_grid_size(self, param_grid):
        total_combos = 1
        for grid_edge in param_grid.values():
            total_combos *= len(grid_edge)
        return total_combos

    def generate_sweep_data(self, param_grid, output_path):
        column_names = list(param_grid.keys()) + self.clip_names
        rows = []
        param_grid_size = self.calc_param_grid_size(param_grid)
        print(f'generating sweep data for {param_grid_size} unique parameter sets')
        for param_set in tqdm(product(*list(param_grid.values())), total=param_grid_size):
            config = SimpleNamespace(**{k: v for k, v in list(zip(param_grid.keys(), param_set))})
            config.behavior_check_window = 60
            br = BehaviorRecognizer(config)
            framegrab_step = int(config.framegrab_interval * 30)
            activity_fractions = []
            for clip_name in self.clip_names:
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

    def execute_sweep(self, input_path, output_path, train_fraction=0.5):
        print(f'executing sweep on {input_path.name}')
        train_ids, test_ids = train_test_split(self.clip_names, train_size=train_fraction, random_state=42)
        y_train, y_test = [np.array([self.label_to_int_mapping[self.clipname_to_label_mapping[cn]] for cn in split])
                           for split in [train_ids, test_ids]]

        df = pd.read_hdf(input_path)
        data_df = df[self.clip_names]
        model = LogisticRegression(random_state=0)
        f1_train, f1_test, boundary = [], [], []
        for idx, row in data_df.iterrows():
            X_train, X_test = row[train_ids].values.reshape(-1, 1), row[test_ids].values.reshape(-1, 1)
            model.fit(X_train, y_train)
            f1_train.append(f1_score(y_train, model.predict(X_train)))
            f1_test.append(f1_score(y_test, model.predict(X_test)))
            boundary.append(np.nan if model.coef_[0][0] == 0 else -model.intercept_[0]/model.coef_[0][0])
        output_df = df[[cname for cname in df.columns if cname not in self.clip_names]].copy()
        output_df['f1_test'] = f1_test
        output_df['f1_train'] = f1_train
        output_df['boundary'] = boundary
        output_df.to_hdf(output_path, key='df')
        print('done')

    def summarize_sweep_results(self, input_path, output_path):
        fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5))
        data = pd.read_hdf(input_path)

        sns.scatterplot(data, x='ooi_confidence_thresh', y='f1_test', ax=axes[0][0])
        sns.scatterplot(data, x='framegrab_interval', y='f1_test', ax=axes[0][1])
        sns.scatterplot(data, x='behavior_max_gap_within_event', y='f1_test', ax=axes[1][0])
        sns.scatterplot(data, x='behavior_min_event_length', y='f1_test', ax=axes[1][1])

        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)

    def generate_dbscan_effect_visualiztion(self, clip_name, output_path):

        occupancies = self.convert_dets_to_occupancy(clip_name, 0.5)
        occupancies['timestamp'] = occupancies.index / 30
        data_slice = occupancies.query('occupancy==2')
        clustering_target = data_slice['timestamp'].values
        labels = DBSCAN1D(0.5, 10).fit_predict(clustering_target)
        labels = pd.Series(data=labels, index=data_slice.index).reindex(occupancies.index, fill_value=-1)
        for lid in labels.unique():
            if lid != -1:
                start = labels[labels == lid].index.min()
                stop = labels[labels == lid].index.max()
                labels.loc[start:stop] = lid
        labels = 2 * (labels >= 0).astype(int)
        fig, axes = fig, ax = plt.subplots(2, 1, figsize=(13.33, 6), sharey=True)
        axes[0].plot(occupancies.timestamp, occupancies.occupancy)
        axes[0].set(title='original occupancies', ylabel='occupancy', xlabel='time (s)')
        axes[1].plot(occupancies.timestamp, labels)
        axes[1].set(title='occupancies filtered with DBSCAN1D', ylabel='occupancy', xlabel='time (s)')
        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)


clustering_param_grid = {
    'ooi_confidence_thresh': np.arange(0.1, 1.0, 0.1),
    'framegrab_interval': np.array([1, 6, 8, 10, 15, 30]) / 30,
    'behavior_max_gap_within_event': np.arange(0, 5.5, 0.5),
    'behavior_min_event_length': np.arange(0, 10, 1),
    'behavior_min_individuals_roi': [2],
    'behavior_max_individuals_roi': [2],
    'behavior_use_dbscan': [True]
}

nonclustering_param_grid = {
    'ooi_confidence_thresh': np.arange(0.1, 1.0, 0.1),
    'framegrab_interval': np.array([1, 6, 8, 10, 15, 30]) / 30,
    'behavior_max_gap_within_event': [0],
    'behavior_min_event_length': [0],
    'behavior_min_individuals_roi': [2],
    'behavior_max_individuals_roi': [2],
    'behavior_use_dbscan': [False]
}


my_base_dir = pathlib.Path(r"C:\Users\tucke\PycharmProjects\InternetOfFish2.0\projects\benchmarking")
my_labeled_data_dir = my_base_dir / 'labeled_data'
results_dir = my_base_dir / 'results'
# sweeper = ParameterSweeper(my_labeled_data_dir)
# sweeper.summarize_sweep_results(results_dir / 'nonclustering_results.h5', results_dir/'nonclustering_results.pdf')
# sweeper.summarize_sweep_results(results_dir / 'clustering_results.h5', results_dir/'clustering_results.pdf')
# sweeper.generate_dbscan_effect_visualiztion(sweeper.clip_names[-1], results_dir / 'dbscan_vis.pdf')
# sweeper.generate_data_summary(results_dir / 'data_summary_30fps.pdf')
# sweeper.generate_data_summary(results_dir / 'data_summary_4fps.pdf', frame_interval=8)
#
# sweeper.generate_sweep_data(nonclustering_param_grid, results_dir / 'nonclustering_sweep.h5')
# sweeper.execute_sweep(results_dir / 'nonclustering_sweep.h5', results_dir / 'nonclustering_results.h5')
# sweeper = ParameterSweeper(data_dir)

# sweeper.generate_sweep_data(clustering_param_grid, results_dir / 'clustering_sweep.h5')
# sweeper.execute_sweep(results_dir / 'clustering_sweep.h5', results_dir / 'clustering_results.h5')


