from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from ..yolo.data import get_feed_vals
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
    meta = self.meta
    labels = meta['labels']
    H, W, _ = meta['out_size']
    B = meta['num']
    C = meta['classes']
    anchors = meta['anchors']

    return get_feed_vals(H, W, C, B, labels)


