from typing import List, Dict
import numpy as np
from beagles.base import PreprocessedBox, Image32

def box_constructor(meta: Dict,
                    net_out_in: Image32,
                    ) -> List[PreprocessedBox]:
    pass