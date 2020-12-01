import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from beagles.base.box import PreprocessedBox

#OVERLAP
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_overlap_c(float x1, float d1 , float x2 , float d2):
    cdef:
        float l1,l2,left,right
    l1 = x1 - d1 / 2.
    l2 = x2 - d2 / 2.
    left = max(l1,l2)
    r1 = x1 + d1 / 2.
    r2 = x2 + d2 / 2.
    right = min(r1, r2)
    return right - left

#BOX INTERSECTION
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_intersection_c(float a_x, float a_y, float a_w, float a_h, float b_x, float b_y, float b_w, float b_h):
    cdef:
        float w,h,area
    w = box_overlap_c(a_x, a_w, b_x, b_w)
    h = box_overlap_c(a_y, a_h, b_y, b_h)
    if w < 0 or h < 0: return 0
    area = w * h
    return area

def intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    return box_intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)

#BOX UNION
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_union_c(float a_x, float a_y, float a_w, float a_h, float b_x, float b_y, float b_w, float b_h):
    cdef:
        float i,u,w,h,area
    i = box_intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)
    u = a_w * a_h + b_w * b_h - i
    return u

#BOX IOU
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(float a_x, float a_y, float a_w, float a_h, float b_x, float b_y, float b_w, float b_h):
    return box_intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h) / box_union_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)

#NMS
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef nms(float[:, ::1] final_probs , float[:, ::1] final_bbox):

    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,i,j,k

    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    boxes = list()

    for i in range(class_length):
        for j in range(pred_length):
            if final_probs[j,i] == 0: 
               continue
            for k in range(j+1, pred_length):
                if final_probs[k,i] == 0: 
                    continue
                if j == k :
                    continue
                if box_iou_c(final_bbox[j,0],final_bbox[j,1],final_bbox[j,2],final_bbox[j,3],final_bbox[k,0],final_bbox[k,1],final_bbox[k,2],final_bbox[k,3]) >= 0.4:
                    if final_probs[k,i] > final_probs[j,i]:
                        final_probs[j,i] = 0
                        break
                    final_probs[k,i] = 0
            if j not in indices:
                bb = PreprocessedBox(x=final_bbox[j, 0], y=final_bbox[j, 1],
                                     w=final_bbox[j, 2], h=final_bbox[j, 3],
                                     c=final_bbox[j, 4],
                                     probs=np.asarray(final_probs[j,:]))
                boxes.append(bb)
                indices.add(j)
    return boxes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef soft_nms(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,i,j,k


    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for i in range(class_length):
        # first box
        for j in range(pred_length):
            if final_probs[j,i] == 0:
                continue
            # second box
            for k in range(j + 1, pred_length):
                if final_probs[k,i] == 0:
                    continue
                if j==k :
                    continue
                else:
                    final_probs[j, i] = final_probs[j, i] * box_iou_c(final_bbox[j,0],final_bbox[j,1],final_bbox[j,2],final_bbox[j,3],final_bbox[k,0],final_bbox[k,1],final_bbox[k,2],final_bbox[k,3])
                    final_probs[k, i] = final_probs[k, i] * box_iou_c(final_bbox[j,0],final_bbox[j,1],final_bbox[j,2],final_bbox[j,3],final_bbox[k,0],final_bbox[k,1],final_bbox[k,2],final_bbox[k,3])
                    break

            if j or k not in indices:
                for l in [j, k]:
                    bb = PreprocessedBox(x=final_bbox[l, 0],
                                         y=final_bbox[l, 1],
                                         w=final_bbox[l, 2],
                                         h=final_bbox[l, 3],
                                         c=final_bbox[l, 4],
                                         probs=np.asarray(final_probs[l,:]))
                    boxes.append(bb)
                    indices.add(l)
    return boxes