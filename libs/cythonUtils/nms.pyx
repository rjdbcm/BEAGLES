import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libs.utils.box import BoundingBox

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
        float i,u
    i = box_intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)
    u = a_w * a_h + b_w * b_h -i
    return u

def union_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    return box_union_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)

#BOX IOU
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(float a_x, float a_y, float a_w, float a_h, float b_x, float b_y, float b_w, float b_h):
    return box_intersection_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h) / box_union_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)

def iou_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    return box_iou_c(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)

#NMS
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef nms(float[:, ::1] final_probs , float[:, ::1] final_bbox):

    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2


    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        for index in range(pred_length):
            if final_probs[index,class_loop] == 0: continue
            for index2 in range(index+1,pred_length):
                if final_probs[index2,class_loop] == 0: continue
                if index==index2 : continue
                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.4:
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] =0
                        break
                    final_probs[index2,class_loop]=0
            
            if index not in indices:
                bb=BoundingBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.c = final_bbox[index, 4]
                bb.probs = np.asarray(final_probs[index,:])
                boxes.append(bb)
                indices.add(index)
    return boxes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef soft_nms(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2


    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        # first box
        for index in range(pred_length):
            if final_probs[index,class_loop] == 0:
                continue
            # second box
            for index2 in range(index + 1, pred_length):
                if final_probs[index2,class_loop] == 0:
                    continue
                if index==index2 :
                    continue
                else:
                    final_probs[index, class_loop] = final_probs[index, class_loop] * box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3])
                    final_probs[index2, class_loop] = final_probs[index2, class_loop] * box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3])
                    break

            if index or index2 not in indices:
                bb=BoundingBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.c = final_bbox[index, 4]
                bb=BoundingBox(class_length)
                bb.x = final_bbox[index2, 0]
                bb.y = final_bbox[index2, 1]
                bb.w = final_bbox[index2, 2]
                bb.h = final_bbox[index2, 3]
                bb.c = final_bbox[index2, 4]
                boxes.append(final_probs[index, class_loop])
                boxes.append(final_probs[index2, class_loop])
                indices.add(index)
                indices.add(index2)
    return boxes