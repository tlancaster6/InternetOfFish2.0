

def test_roi_detection(roi_detector_fixture, sample_img_loader_fixture):
    roid, img = roi_detector_fixture, sample_img_loader_fixture
    bbox = roid.detect_effdet(img)[0].bbox
    assert bbox.xmin + bbox.xmax + bbox.ymin + bbox.ymax == 1971


def test_ooi_detection(ooi_detector_fixture, sample_img_loader_fixture):
    ooid, img = ooi_detector_fixture, sample_img_loader_fixture
    bbox = ooid.detect_effdet(img)
    assert bbox.xmin + bbox.xmax + bbox.ymin + bbox.ymax == 1971