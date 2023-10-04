import pandas as pd
import cv2
import pathlib
import numpy as np
import sys

FILE = pathlib.Path(__file__).resolve()
MODULE_DIR = FILE.parent  # repository root
REPO_ROOT_DIR = MODULE_DIR.parent
MODEL_DIR = REPO_ROOT_DIR / 'models'
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))

from modules.object_detection import DetectorBase


def generate_dense_detection_data(video_path: pathlib.Path):
    print(f'processing {video_path.name}')
    ooid = DetectorBase(MODEL_DIR / 'ooi.tflite', confidence_thresh=0.1)
    cap = cv2.VideoCapture(str(video_path))
    current_frame = 0
    rows = []
    columns = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'score']
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = ooid.detect(img)
        if not dets:
            rows.append([current_frame] + ([np.nan] * 5))
        for det in dets:
            rows.append([current_frame, det.bbox.xmin, det.bbox.xmax, det.bbox.ymin, det.bbox.ymax, det.score])
        current_frame += 1
    cap.release()
    df = pd.DataFrame(rows, columns=columns).set_index('frame')
    df.to_csv(str(video_path.with_suffix('.csv')))
