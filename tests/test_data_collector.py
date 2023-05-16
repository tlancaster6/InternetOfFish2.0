import pytest
import pathlib
import shutil
from time import sleep
from modules.data_collection import DataCollector


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


def test_init_camera(data_collector_fixture):
    dc = data_collector_fixture
    actual_params = {'framerate': dc.cam.framerate, 'resolution': dc.cam.resolution}
    assert dc.picamera_kwargs == actual_params


def test_collect_video(data_collector_fixture):
    dc = data_collector_fixture
    dc.start_recording()
    sleep(2)
    dc.stop_recording()
    assert dc.video_dir.glob('*.h264')



