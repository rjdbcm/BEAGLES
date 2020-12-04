import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from beagles.utils.box import BoundingBox
from nms cimport NMS

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef logit_c(float x):
    cdef float y = 1/(1 + exp(x))
    return y

# noinspection DuplicatedCode
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def box_constructor(meta,np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2
        float truth_thresh = meta['truth_thresh']
        float ignore_thresh = meta['ignore_thresh']
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']

    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)

