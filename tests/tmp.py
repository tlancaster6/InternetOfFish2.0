from tests.mocks import MockDataCollector
from modules.object_detection import DetectorBase
import pathlib
import cv2


TESTING_RESOURCE_DIR = pathlib.Path(__file__).resolve().parents[0] / 'resources'
mdc = MockDataCollector(TESTING_RESOURCE_DIR / 'sample_clip.mp4')
roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')
ooid = DetectorBase(TESTING_RESOURCE_DIR / 'ooi.tflite')
img = cv2.cvtColor(cv2.imread(TESTING_RESOURCE_DIR / 'sample_image.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img_cropped = cv2.cvtColor(cv2.imread(TESTING_RESOURCE_DIR / 'sample_image_cropped.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
roid_dets = roid.detect(img)
ooid_dets = ooid.detect(img_cropped)
