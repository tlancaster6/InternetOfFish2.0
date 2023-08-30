import pytest
import pathlib
import shutil
import cv2
import sys
from modules.data_collection import DataCollector
from modules.object_detection import DetectorBase
from tests.mocks import MockDataCollector
import os

FILE = pathlib.Path(__file__).resolve()
TESTING_DIR = FILE.parents[0]
REPO_ROOT_DIR = TESTING_DIR.parents[0]  # repository root
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))  # add REPO_ROOT_DIR to system path
TESTING_RESOURCE_DIR = TESTING_DIR / 'resources'
MODEL_DIR = REPO_ROOT_DIR / 'models'
DATA_DIR = REPO_ROOT_DIR / 'projects'
SENDGRID_CREDENTIAL_FILE = REPO_ROOT_DIR / 'credentials' / 'sendgrid_key.secret'


@pytest.fixture
def tmp_dir_fixture():
    tmp = pathlib.Path('./tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def data_collector_fixture(tmp_dir_fixture):
    picamera_kwargs = {'framerate': 30, 'resolution': (1280, 976)}
    dc = DataCollector(tmp_dir_fixture, picamera_kwargs)
    yield dc
    dc.shutdown()


@pytest.fixture
def mock_data_collector_fixture():
    mock_dc = MockDataCollector(TESTING_RESOURCE_DIR / 'sample_clip.mp4')
    yield mock_dc
    mock_dc.shutdown()


@pytest.fixture
def sample_img_loader_fixture():
    img = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image.png'), cv2.IMREAD_COLOR)
    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def sample_cropped_img_loader_fixture():
    img = cv2.imread(str(TESTING_RESOURCE_DIR / 'sample_image_cropped.png'), cv2.IMREAD_COLOR)
    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def roi_detector_fixture():
    yield DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')


@pytest.fixture
def ooi_detector_fixture():
    yield DetectorBase(TESTING_RESOURCE_DIR / 'ooi.tflite')
