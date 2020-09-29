from collections import namedtuple
from typing import List, Union, NamedTuple
from abc import abstractmethod
import numpy as np


class PostprocessedBox(NamedTuple):
    """
    Holds labeled bounding box returned by framework's `postprocess` method.
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    difficult: bool


class PreprocessedBox(NamedTuple):
    """
    Holds a box returned from a cython box constructor
    """
    x: int
    y: int
    w: int
    h: int
    c: int
    probs: np.ndarray


class ProcessedBox(NamedTuple):
    """
    Holds a box returned by a backend framework's `process_box` and `findboxes` methods.
    """
    left: float
    right: float
    top: float
    bot: float
    label: str
    max_idx: int
    max_prob: float
