def test_mock_data_collector(mock_data_collector_fixture):
   mdc = mock_data_collector_fixture
   print(mdc.resolution)
   print(mdc.framerate)
   assert True

#
# def test_roi_detection(mock_data_collector_fixture, roi_detector_fixture):
