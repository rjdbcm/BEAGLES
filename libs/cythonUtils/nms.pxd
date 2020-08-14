import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from libs.utils.box import BoundingBox


cdef nms(float[:, ::1] , float[:, ::1])

cdef soft_nms(float[:, ::1], float[:, ::1])

