"""Backend IO operations"""
from beagles.backend.io.config_parser import *
from beagles.backend.io.pascal_voc_clean_xml import *

ConfigParser = ConfigParser
"""
:class:`ConfigParser`
"""

DarknetConfigFile = DarknetConfigFile
"""
:class:`DarknetConfigFile`
"""

#
# CheckpointLoader = CheckpointLoader
# """
# :class:`CheckpointLoader`
# """
#
# WeightsLoader = WeightsLoader
# """
# :class:`WeightsLoader`
# """

pascal_voc_clean_xml = pascal_voc_clean_xml
"""
Parses PASCAL VOC XML annotations
"""