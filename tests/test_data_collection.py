from time import sleep


def test_init_camera(data_collector_fixture):
    dc = data_collector_fixture
    actual_params = {'framerate': dc.cam.framerate, 'resolution': dc.cam.resolution}
    assert dc.picamera_kwargs == actual_params


def test_collect_video(data_collector_fixture):
    dc = data_collector_fixture
    dc.start_recording()
    sleep(2)
    dc.stop_recording()
    assert list(dc.video_dir.glob('*.h264'))


def test_split_recording(data_collector_fixture):
    dc = data_collector_fixture
    dc.start_recording()
    sleep(2)
    dc.split_recording()
    sleep(2)
    dc.stop_recording()
    assert len(list(dc.video_dir.glob('*.h264'))) == 2


def test_capture_frame(data_collector_fixture):
    dc = data_collector_fixture
    dc.start_recording()
    sleep(2)
    img = dc.capture_frame()
    dc.stop_recording()
    assert img.sum()


def test_mock_data_collector(mock_data_collector_fixture):
    mdc = mock_data_collector_fixture
    img = mdc.capture_frame()
    assert img.sum() == 309161381
