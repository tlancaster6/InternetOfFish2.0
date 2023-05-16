import pytest
import pathlib
import shutil
from modules.data_collection import DataCollector


@pytest.fixture
def tmp_dir_fixture():
    tmp = pathlib.Path('./tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_init_camera(tmp_dir_fixture):
    expected_params = {'framerate': 30, 'resolution': (1280, 976)}
    dc = DataCollector(tmp_dir_fixture, expected_params)
    actual_params = {'framerate': dc.cam.framerate, 'resolution': dc.cam.resolution}
    assert expected_params == actual_params





