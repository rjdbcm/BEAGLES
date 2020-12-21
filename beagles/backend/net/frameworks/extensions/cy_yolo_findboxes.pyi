from beagles.base import PreprocessedBox, Image32
from typing import List, Dict

def box_constructor(meta: Dict,
                    net_out: Image32,
                    threshold: float
                    ) -> List[PreprocessedBox]:
    pass