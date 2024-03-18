import cv2
import pathlib
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter, run_inference
import pandas as pd

FILE = pathlib.Path(__file__).resolve()
REPO_ROOT_DIR = FILE.parent  # repository root
MODEL_DIR = REPO_ROOT_DIR / 'models'
RESOURCE_DIR = REPO_ROOT_DIR / 'resources'
HOME_DIR = pathlib.Path().home().resolve()
PICS_DIR = REPO_ROOT_DIR / 'pics'


def read_img(img_path):
    """
    read an image from file and convert it to the format expected by DetectorBase.detect()
    :param img_path: full path to image (str or pathlike)
    :return: Image object
    """
    return cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class DetectorBase:

    def __init__(self, model_path):
        """
        Base class for performing inference
        :param model_path: full path to tflite model file
        """
        self.model_path = model_path
        self.interpreter = make_interpreter(str(model_path))
        self.interpreter.allocate_tensors()
        self.input_size = common.input_size(self.interpreter)

    def detect(self, img, confidence_thresh=0.1):
        """
        perform detection on a single image
        :param img: opencv Image object as returned by the read_img function
        :param confidence_thresh: detections with confidence below this threshold will be discarded from the return
        :return: list of detection objects, sorted from high confidence to low confidence
        """
        scale = (self.input_size[1] / img.shape[1], self.input_size[0] / img.shape[0])
        img = cv2.resize(img, self.input_size)
        run_inference(self.interpreter, img.tobytes())
        dets = detect.get_objects(self.interpreter, confidence_thresh, scale)
        return sorted(dets, reverse=True, key=lambda x: x.score)

    def detect_multiple(self, img_dir, confidence_thresh=0.1, outfile_name='detections.csv'):
        """
        run detection on all images in a directory and save the results as a formatted csv
        :param img_dir: full path to image directory
        :param confidence_thresh: see DetectorBase.detect()
        :param outfile_name: file name for the output csv (including .csv extension)
        :return: formatted dataframe of all detections (same dataframe as saved to csv)
        """
        img_dir = pathlib.Path(img_dir)
        img_paths = list(img_dir.glob('*.png'))
        df = []
        for ip in img_paths:
            dets = self.detect(read_img(ip), confidence_thresh)
            for det in dets:
                df.append([ip.name, det.bbox.xmin, det.bbox.xmax, det.bbox.ymin, det.bbox.ymax, det.score])
        columns = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'score']
        df = pd.DataFrame(df, columns=columns)
        df.to_csv(str(img_dir / outfile_name))
        return df

fish_model_path = MODEL_DIR / 'effdet0_fish' / 'ooi.tflite'
fish_detector = DetectorBase(fish_model_path)
img_path = RESOURCE_DIR / 'sample_image_cropped.png'
img = read_img(img_path)
detections = fish_detector.detect(img)
# df = fish_detector.detect_multiple(HOME_DIR / 'testing')
df = fish_detector.detect_multiple(PICS_DIR, confidence_thresh = 0.01)