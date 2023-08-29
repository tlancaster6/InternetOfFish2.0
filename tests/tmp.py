from tests.mocks import MockDataCollector
from modules.object_detection import DetectorBase
import pathlib


TESTING_RESOURCE_DIR = pathlib.Path(__file__).resolve().parents[0] / 'resources'
mdc = MockDataCollector(TESTING_RESOURCE_DIR / 'sample_clip.mp4')
roid = DetectorBase(TESTING_RESOURCE_DIR / 'roi.tflite')


