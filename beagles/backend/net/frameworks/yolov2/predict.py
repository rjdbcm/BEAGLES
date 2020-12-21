import numpy as np
from beagles.backend.net.frameworks.extensions.cy_yolo2_findboxes import box_constructor

def find(self, net_out):
    if type(net_out) is not np.ndarray:
        net_out = np.asarray(net_out)
    if net_out.ndim == 4:
        net_out = np.concatenate(net_out, 0)
    return box_constructor(self.meta, net_out) or []
