cimport numpy as np

ctypedef np.float_t DTYPE_t

cdef nms(float[:, ::1] , float[:, ::1])

cdef soft_nms(float[:, ::1], float[:, ::1])

