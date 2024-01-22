import pathlib
import sys
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from statistics import mean, median, stdev
import pandas as pd
from modules.object_detection import DetectorBase
from modules.utils import generate_dense_detection_data
import numpy as np
import xml.etree.ElementTree as ET

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


class BenchmarkerBase:

    def __init__(self, model_id):
        self.model_id = model_id
        self.log_dir = LOG_DIR / model_id
        if not self.log_dir.exists():
            self.log_dir.mkdir()


class ObjectDetectionBenchMarker(BenchmarkerBase):
    def __init__(self, model_id):
        super().__init__(model_id)

    def run_all_benchmarks(self):
        print('benchmarking ooi detection speed')
        self.benchmark_ooid_speed()
        print('benchmarking roi detection speed')
        self.benchmark_roid_speed()
        print('benchmarking ooi and roi detection under realistic operating conditions')
        self.realistic_benchmark()
        print('done')

    def benchmark_detection_speed(self, detector_object, img, reps=500):
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
        roid = DetectorBase(MODEL_DIR / self.model_id / 'roi.tflite')
        img = cv2.cvtColor(cv2.imread(str(RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        metrics = self.benchmark_detection_speed(roid, img)
        metrics.to_csv(str(self.log_dir / 'roi_speed_benchmark.log'))

    def benchmark_ooid_speed(self):
        ooid = DetectorBase(MODEL_DIR / self.model_id / 'ooi.tflite')
        img = cv2.cvtColor(cv2.imread(str(RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        metrics = self.benchmark_detection_speed(ooid, img)
        metrics.to_csv(str(self.log_dir / 'ooi_speed_benchmark.log'))

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
        roid = DetectorBase(MODEL_DIR / self.model_id / 'roi.tflite')
        ooid = DetectorBase(MODEL_DIR / self.model_id / 'ooi.tflite')

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
        ooid_metrics.to_csv(str(self.log_dir / 'ooi_speed_benchmark2.log'))
        roid_metrics.to_csv(str(self.log_dir / 'roi_speed_benchmark2.log'))

        # clean up
        shutil.rmtree(tmp_dir)



class OccupancyAccuracyBenchmarker(BenchmarkerBase):

    def __init__(self, model_id, validation_data_dir):
        self.validation_data_dir = validation_data_dir
        super().__init__(model_id)

    def test_occupancy_accuracy(self, conf_thresh=0.5):
        val_data_dir = pathlib.Path(self.validation_data_dir)
        if not val_data_dir.exists():
            print('cannot locate data for occupancy accuracy testing')
            return
        ooid = DetectorBase(MODEL_DIR / self.model_id / 'ooi.tflite')
        img_paths = list(val_data_dir.glob('*.jpg'))
        n_correct = 0
        total_squared_error = 0
        total_signed_error = 0
        for ip in img_paths:
            xml_path = ip.with_suffix('.xml')
            n_fish_actual = len(ET.parse(str(xml_path)).getroot().findall('./object'))
            img = cv2.cvtColor(cv2.imread(str(ip), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            n_fish_pred = len([x for x in ooid.detect(img) if x.score >= conf_thresh])
            if n_fish_pred == n_fish_actual:
                n_correct += 1
            total_squared_error += (n_fish_pred - n_fish_actual) ** 2
            total_signed_error += n_fish_pred - n_fish_actual
        accuracy = n_correct / len(img_paths)
        rms_error = (total_squared_error / len(img_paths)) ** 0.5
        avg_signed_error = total_signed_error / len(img_paths)
        output = pd.Series({'conf_thresh': conf_thresh,
                            'accuracy': accuracy,
                            'rms_error': rms_error,
                            'avg_signed_error': avg_signed_error})
        output.to_csv(str(self.log_dir / 'occupancy_accuracy.log'))


class BehaviorDetectionBenchmarker(BenchmarkerBase):

    def __init__(self, model_id, annotated_clip_dir):
        super().__init__(model_id)
        self.clip_dir = pathlib.Path(annotated_clip_dir)
        self.data_paths = list(self.clip_dir.glob('**/*.csv'))
        self.clipname_to_label_mapping = self.map_clipnames_to_labels()
        self.clip_names = list(self.clipname_to_label_mapping.keys())
        self.label_to_int_mapping = self.map_labels_to_ints()
        self.data = self.read_data()

    def generate_dense_detection_data(self):
        vid_paths = pathlib.Path(self.clip_dir).glob('**/*.mp4')
        for vp in vid_paths:
            print(f'generating dense detection data for {vp.name}')
            generate_dense_detection_data(vp, self.model_id)
        print('Done')
        self.data = self.read_data()

    def read_data(self):
        if not self.data_paths:
            return
        df = []
        for dp in self.data_paths:
            tmp = pd.read_csv(str(dp), index_col=0)
            tmp['clip_name'] = dp.stem
            tmp['label'] = self.clipname_to_label_mapping[dp.stem]
            df.append(tmp)
        return pd.concat(df)

    def map_labels_to_ints(self):
        string_labels = pd.Series([dp.parent.name for dp in self.data_paths]).unique()
        int_labels = np.arange(len(string_labels))
        return {k: v for k, v in list(zip(string_labels, int_labels))}

    def map_clipnames_to_labels(self):
        return {dp.stem: dp.parent.name for dp in self.data_paths}

    def convert_dets_to_occupancy(self, clip_name: str, confidence_thresh: float):
        occupancy = self.data.query('clip_name==@clip_name')
        occupancy = (occupancy['score'] >= confidence_thresh).groupby(level=0).sum()
        occupancy.name = 'occupancy'
        return pd.DataFrame(occupancy)

    def generate_data_summary(self, frame_interval=1):
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
        axes[1][0].set(title='distribution of double occupancy fractions\n score>=0.5')

        data = []
        for cn in self.clip_names:
            label = self.clipname_to_label_mapping[cn]
            occupancy = self.convert_dets_to_occupancy(cn, 0.9)
            occupancy_fraction = occupancy.value_counts(normalize=True)[2] if 2 in occupancy.values else 0.0
            data.append([label, occupancy_fraction])
        data = pd.DataFrame(data, columns=['label', 'double_occupancy_fraction'])
        sns.histplot(data, x='double_occupancy_fraction', hue='label', ax=axes[1][1], bins=20)
        axes[1][1].set(title='distribution of double occupancy fractions\n score>=0.75')

        fig.suptitle('Summary of Benchmarking/Tuning Dataset')
        fig.tight_layout()
        fig.savefig(str(self.log_dir / f'DataSummary_frameinterval{frame_interval}'))
        plt.close(fig)


