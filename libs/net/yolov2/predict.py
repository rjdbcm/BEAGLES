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
from ...pascal_voc_io import PascalVocWriter, XML_EXT


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
    Args:
        net_out: A single fetch from tf session.run
        im: A path or pathlike object to an image file
        save: A boolean. Whether to save predictions to disk
            Default True
    Returns:
        imgcv: An annotated np.ndarray if save == False
        or
        None if salve
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
    h, w, c = imgcv.shape

    writer = PascalVocWriter(self.flags.img_out, im, [h, w, c])
    resultsForJSON = []
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        if self.flags.output_type:
            resultsForJSON.append({"label": mess,
                                   "confidence": float('%.2f' % confidence),
                                   "topleft": {"x": left, "y": top},
                                   "bottomright": {"x": right, "y": bot}})
            writer.addBndBox(left, bot, right, top, mess, False)
            #continue

        mess = mess + " " + str(round(confidence, 3))
        cv2.rectangle(imgcv, (left, top), (right, bot), colors[max_indx], 3)
        cv2.putText(
            imgcv, mess, (left, top - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
            (0, 230, 0), 1, cv2.LINE_AA)
    if not save:
        return imgcv

    img_name = os.path.join(self.flags.imgdir, os.path.basename(im))
    if "json" in self.flags.output_type:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)

    if "voc" in self.flags.output_type:
        writer.save(os.path.join(self.flags.imgdir,
                                 os.path.splitext(os.path.basename(im))[
                                     0]+ XML_EXT))
    # uncomment to write annotated images
    # cv2.imwrite(img_name + "out.jpg", imgcv)


