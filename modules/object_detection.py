"""code for running light-weight object detection to locate animals and regions of interest (roi's)"""
import os, logging, time
from collections import namedtuple

from glob import glob
import cv2
from math import sqrt
import numpy as np

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter, run_inference


class DetectorBase:

    def __init__(self, model_path, confidence_thresh=0.25):
        self.confidence_thresh = confidence_thresh
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.inference_size = common.input_size(self.interpreter)

    def detect(self, img):
        scale = (self.inference_size[1] / img.shape[1], self.inference_size[0] / img.shape[0])
        img = cv2.resize(img, self.inference_size)
        run_inference(self.interpreter, img.tobytes())
        dets = detect.get_objects(self.interpreter, self.confidence_thresh, scale)
        return sorted(dets, reverse=True, key=lambda x: x.score)









