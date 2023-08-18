def test_mock_data_collector(mock_data_collector_fixture):
    mdc = mock_data_collector_fixture
    img = mdc.capture_frame()
    assert mdc.resolution == (img.shape[1], img.shape[0])


def test_roi_detection(mock_data_collector_fixture, roi_detector_fixture):
    mdc, roid = mock_data_collector_fixture, roi_detector_fixture
    img = mdc.capture_frame()
    dets = roid.detect_effdet(img)


