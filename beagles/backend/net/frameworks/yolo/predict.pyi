from typing import List, Union, Any, Tuple, SupportsFloat, SupportsInt, B
from os import PathLike
import numpy as np
from nptyping import Float32, NDArray
from beagles.base.typing import Image8, Image32
from beagles.base.box import PreprocessedBox, ProcessedBox


def resize_input(self,
                 image: Image8) -> Image32:
    """
    Resizes/reformats image and casts to float32

    Args:
        self: A Framework Subsystem

        image: array of pixels shape(rows,columns,channels) where channels have dtype int8 and BGR ordering

    Returns:
        array of pixels shape(rows,columns,channels) where channels have dtype float32 and RGB ordering

    """
    pass

def process(self,
            b: PreprocessedBox,
            w: SupportsInt,
            h: SupportsInt,
            threshold: SupportsFloat) -> ProcessedBox:
    pass

def find(self,
        net_out: NDArray[Float32]) -> List[PreprocessedBox]:
    pass

def preprocess(self,
               image: Union[Image8, PathLike],
               allobj: List = None) -> Tuple[Image32, Union[List, None]]:
    """
    Takes an image, return it as a numpy ndarray that is readily
    to be fed into a tensorflow graph. Expects an RGB colorspace for augmentations.

    Note:
        If there is an accompanied annotation (allobj),
        meaning this preprocessing is being used for training, then this
        image and accompanying bounding boxes will be transformed.

    Args:
        self: A Framework Subsystem

        image: An np.ndarray or file-like image object.

        allobj: List of annotated objects.

    Returns (if allobj == None):
        image: A resized np.float32 np.ndarray representation of the image

    Returns (if allobj != None):
        image: A randomly transformed and recolored np.float32 np.ndarray
        bboxes: Transformed bounding boxes
    """
    pass

def postprocess(self,
                net_out: NDArray[Float32],
                image: PathLike,
                save: bool = True) -> Union[Image32, None]:
    """
    Takes net output, draw predictions, saves to disk turns :class:`ProcessedBox` into :class:`PostprocessedBox`

    Args:
        self: A Framework Subsystem

        net_out: Raw neural net output as float32 np.ndarray

        image: A path or pathlike object to an image file.

        save: Whether to save predictions to disk defaults to True.

    Returns:
        imgcv: An annotated np.ndarray if save == False
        or
        None if save == True
    """
    pass