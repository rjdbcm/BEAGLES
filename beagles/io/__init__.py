from beagles.io.flags import *
from beagles.io.sharedmemory import *
from beagles.io.logs import *
from beagles.io.obs import *
from beagles.io.yolo import *
from beagles.io.labelFile import *
from beagles.io.settings import *
from beagles.io.pascalVoc import *


# noinspection PyUnresolvedReferences
__all__ = ['DO_NOT_WILD_IMPORT']

SharedMemory = SharedMemory
"""
:class:`SharedMemory`
"""

TiledCaptureArray = TiledCaptureArray
"""
:class:`TiledCaptureArray`
"""

LabelFile = LabelFile
"""
:class:`LabelFile`
"""

YoloReader = YoloReader
"""
:class:`YoloReader`
"""

YoloWriter = YoloWriter
"""
:class:`YoloWriter`
"""

PascalVocReader = PascalVocReader
"""
:class:`PascalVocReader`
"""

PascalVocWriter = PascalVocWriter
"""
:class:`PascalVocWriter`
"""

Settings = Settings
"""
:class:`Settings`
"""

get_logger = get_logger
"""Convenience function to get a :class:`logging.Logger`

Args:
    level (int): One of `logging.FATAL`, `logging.ERROR`, `logging.WARN`, `logging.INFO`, or `logging.DEBUG`.

Return:
    :class:`logging.Logger` with logging level set to `level`

"""

datetime_from_filename = datetime_from_filename
"""Convenience function to get datetime.strptime from filenames containing ISO 8601 timestamps.
    
Extracts a datetime object from OBS Video Filename containing 
%CCYY-%MM-%DD %hh-%mm-%ss or %CCYY-%MM-%DD_%hh-%mm-%ss.

Args:
    filename: File to extract datetime from.
    fmt: Format to return strptime in one of `'underscore'` or `'space'`
    
Returns:
    A strptime string containing the datetime extracted from `filename`

"""