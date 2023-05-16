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
        self.input_size = common.input_size(self.interpreter)

        self.input_details = self.interpreter.get_input_details()
        self.scale, self.zero_point = self.input_details[0]['quantization']
        self.output_details = self.interpreter.get_output_details()

    def detect_yolo(self, img):
        h, w, ch = img.shape
        img = (img / self.scale + self.zero_point).astype(np.uint8)
        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output['index'])
            x = (x.astype(np.float32) - self.zero_point) * self.scale  # re-scale
            y.append(x)
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        y[0][..., :4] *= [w, h, w, h]

    def detect_effdet(self, img):
        scale = (self.input_size[1] / img.shape[1], self.input_size[0] / img.shape[0])
        img = cv2.resize(img, self.input_size)
        run_inference(self.interpreter, img.tobytes())
        dets = detect.get_objects(self.interpreter, self.confidence_thresh, scale)
        return sorted(dets, reverse=True, key=lambda x: x.score)







