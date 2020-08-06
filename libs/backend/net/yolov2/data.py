from libs.utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from ..yolo.data import get_feed_values
from copy import deepcopy
import pickle
import numpy as np
import os


def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    # sizes are passed to avoid having duplicate get_feed_values methods for
    # YOLO and YOLOv2
    H, W, _ = self.meta['out_size']
    return self.get_feed_values(chunk, H, W)


