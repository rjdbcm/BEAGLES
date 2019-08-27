import numpy as np
import math
import sys
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from ...yolo_io import YOLOWriter
from ...pascal_voc_io import PascalVocWriter


def expit(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def findboxes(self, net_out):
    # meta
    meta = self.meta
    boxes = list()
    boxes = box_constructor(meta, net_out)
    return boxes


def postprocess(self, net_out, im, save=True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.findboxes(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im
    h, w, _ = imgcv.shape

    resultsForJSON = []
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        if self.flags.json:
            resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
            #continue
        mess = mess + " " + str(round(confidence, 3))
        cv2.rectangle(imgcv, (left, top), (right, bot), colors[max_indx], 3)
        cv2.putText(
            imgcv, mess, (left, top - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
            (0, 230, 0), 1, cv2.LINE_AA)
    if not save:
        return imgcv

    outfolder = os.path.join(self.flags.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))
    if self.flags.json:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)
    cv2.imwrite(img_name, imgcv)


