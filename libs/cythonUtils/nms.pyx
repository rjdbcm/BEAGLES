import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libs.utils.box import BoundingBox


cdef class Box:
    cdef public float x,y,w,h,c

    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    @classmethod
    def create(cls, float[:, ::1] final_bbox, np.intp_t index):
        return cls(final_bbox[index, 0], final_bbox[index, 1],
                   final_bbox[index, 2], final_bbox[index, 3],
                   final_bbox[index, 4])


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
cdef float box_intersection_c(Box a, Box b):
    cdef:
        float w,h,area
    w = box_overlap_c(a.x, a.w, b.x, b.w)
    h = box_overlap_c(a.x, a.h, b.x, b.h)
    if w < 0 or h < 0: return 0
    area = w * h
    return area



#BOX UNION
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_union_c(Box a, Box b):
    cdef:
        float i,u
    i = box_intersection_c(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


#BOX IOU
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(Box a, Box b):
    return box_intersection_c(a, b) / box_union_c(a, b)


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
        # first box
        for index in range(pred_length):
            box_a = Box.create(final_bbox, index)
            if final_probs[index, class_loop] == 0:
                # skip zero probability box
                continue
            # second box
            for index2 in range(index + 1, pred_length):
                box_b = Box.create(final_bbox, index2)
                if final_probs[index2, class_loop] == 0:
                    continue
                if index==index2 :
                    continue
                if box_iou_c(box_a, box_b) >= 0.4:
                    if final_probs[index2, class_loop] > final_probs[index, class_loop]:
                        final_probs[index, class_loop] = 0
                        break
                    final_probs[index2, class_loop] = 0

            if index not in indices:
                bb=BoundingBox(class_length)
                bb.x = box_a.x
                bb.y = box_a.y
                bb.w = box_a.w
                bb.h = box_a.h
                bb.c = box_a.c
                bb.probs = np.asarray(final_probs[index, :])
                boxes.append(bb)
                indices.add(index)
    return boxes

# cdef NMS(float[:, ::1] final_probs , float[:, ::1] final_bbox):
#     cdef list boxes = list()
#     cdef:
#         np.intp_t pred_length,class_length,class_loop,index,index2, i, j

  
#     pred_length = final_bbox.shape[0]
#     class_length = final_probs.shape[1]

#     for class_loop in range(class_length):
#         order = np.argsort(final_probs[:,class_loop])[::-1]
#         # First box
#         for i in range(pred_length):
#             index = order[i]
#             if final_probs[index, class_loop] == 0.: 
#                 continue
#             # Second box
#             for j in range(i+1, pred_length):
#                 index2 = order[j]
#                 if box_iou_c(
#                     final_bbox[index,0],final_bbox[index,1],
#                     final_bbox[index,2],final_bbox[index,3],
#                     final_bbox[index2,0],final_bbox[index2,1],
#                     final_bbox[index2,2],final_bbox[index2,3]) >= 0.4:
#                     final_probs[index2, class_loop] = 0.
                    
#             bb = BoundBox(class_length)
#             bb.x = final_bbox[index, 0]
#             bb.y = final_bbox[index, 1]
#             bb.w = final_bbox[index, 2]
#             bb.h = final_bbox[index, 3]
#             bb.c = final_bbox[index, 4]
#             bb.probs = np.asarray(final_probs[index,:])
#             boxes.append(bb)
  
#     return boxes

# #NMS
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef soft_nms(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2

    def assign(Box box):
        bb=BoundingBox(class_length)
        bb.x = box.x
        bb.y = box.y
        bb.w = box.w
        bb.h = box.h
        bb.c = box.c
        bb.probs = np.asarray(final_probs[index,:])
        return bb


    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        # first box
        for index in range(pred_length):
            box_a = Box.create(final_bbox, index)
            if final_probs[index,class_loop] == 0:
                continue
            # second box
            for index2 in range(index + 1, pred_length):
                box_b = Box.create(final_bbox, index2)
                if final_probs[index2,class_loop] == 0:
                    continue
                if index==index2 :
                    continue
                else:
                    final_probs[index, class_loop] = final_probs[index, class_loop] * box_iou_c(box_a, box_b)
                    final_probs[index2, class_loop] = final_probs[index2, class_loop] * box_iou_c(box_a, box_b)
                    break

            if index or index2 not in indices:
                boxes.append(assign(box_a))
                boxes.append(assign(box_b))
                indices.add(index)
                indices.add(index2)
    return boxes