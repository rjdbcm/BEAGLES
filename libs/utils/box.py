from collections import namedtuple
import numpy as np


class BoundingBox:
    """Shared class for backend and frontend bounding boxes"""
    def __init__(self, classes: int = 0):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))
        self.boxlist = []

    def appendBox(self, bounding_box: namedtuple, metadata: namedtuple):
        bndbox = namedtuple('bndbox', 'xmin ymin xmax ymax label difficult')
        bndbox = bndbox(bounding_box.xmin, bounding_box.ymin,
                        bounding_box.xmax, bounding_box.ymax,
                        metadata.label, metadata.difficult)
        self.boxlist.append(bndbox)
