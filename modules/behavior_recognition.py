"""code for defining a relationship between object detection data and one or more behaviors of interest"""
import logging
logger = logging.getLogger(__name__)

class BehaviorRecognizer:

    def __init__(self):

        logger.debug('BehaviorRecognizer initialized')

    def append_dets(self, dets):
        pass

    def check_for_behavior(self):
        pass
