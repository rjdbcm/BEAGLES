from typing import List, Union, Any, SupportsInt, Text
from nptyping import NDArray, Int8, Float32

Image8 = NDArray[(Any, Any, 3), Int8]
Image32 = NDArray[(Any, Any, 3), Float32]
AnnotatedBBox = List[Union[Text, SupportsInt]]
