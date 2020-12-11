import os
import sys
import json
from typing import Union, List, Any, Tuple
from functools import partial
from collections import namedtuple
import cv2
import numpy as np
from beagles.backend.net.augmentation.im_transform import Transform
from beagles.backend.net.frameworks.extensions.cy_yolo_findboxes import yolo_box_constructor
from beagles.io.pascalVoc import PascalVocWriter, XML_EXT
from beagles.base.box import PostprocessedBox, ProcessedBox


def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)


def resize_input(self, im):
    h, w, c = self.meta['inp_size']
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz


def process_box(self, b, h, w, threshold) -> ProcessedBox:
    max_idx = np.argmax(b.probs)
    max_prob = b.probs[max_idx]
    if max_prob > threshold:
        left = int((b.x - b.w / 2.) * w)
        right = int((b.x + b.w / 2.) * w)
        top = int((b.y - b.h / 2.) * h)
        bot = int((b.y + b.h / 2.) * h)
        left = 0 if left < 0 else left
        right = w - 1 if right > w - 1 else right
        top = 0 if top < 0 else top
        bot = h - 1 if bot > h - 1 else bot
        mess = f"{self.meta['labels'][max_idx]}"
        return ProcessedBox(left, right, top, bot, mess, max_idx, max_prob)
    return None


def findboxes(self, net_out):
    meta, flags = self.meta, self.flags
    threshold = flags.threshold

    boxes = []
    boxes = yolo_box_constructor(meta, net_out, threshold)

    return boxes


def preprocess(self, image: Union[np.ndarray, Any], allobj: List = None) -> Tuple[np.ndarray, list]:
    """
    Takes an image, return it as a numpy tensor that is readily
    to be fed into a tensorflow graph. Expects a BGR colorspace for augmentations.

    Note:
        If there is an accompanied annotation (allobj),
        meaning this preprocessing is being used for training, then this
        image and accompanying bounding boxes will be transformed.

    Args:
        image: An np.ndarray or file-like image object.

        allobj: List of annotated objects.

    Returns (if allobj == None):
        image: A randomly transformed and recolored np.ndarray

    Returns (if allobj != None):
        image: A randomly transformed and recolored np.ndarray
        bboxes: Transformed bounding boxes
    """
    bboxes = None
    if type(image) is not np.ndarray:
        image = cv2.imread(image) # BRG

    if allobj is not None:  # in training mode
        image, bboxes = self.transform.spatial(image, allobj)
        transformed = self.transform.pixel(image)
        image = transformed["image"]

    image = self.resize_input(image)

    if allobj is None:
        return image
    else:
        return image, bboxes  # , np.array(im) # for unit testing

def postprocess(self, net_out, im: os.PathLike, save: bool = True) -> np.ndarray:
    """Takes net output, draw predictions, saves to disk
        turns :class:`ProcessedBox` into :class:`PostprocessedBox`

    Args:
        net_out: A single fetch from tf session.run.

        im: A path or pathlike object to an image file.

        save: Whether to save predictions to disk defaults to True.

    Returns:
        imgcv: An annotated np.ndarray if save == False
        or
        None if save == True
    """
    boxes = self.findboxes(np.asarray(net_out))

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
        pb = self.process_box(b, h, w, threshold)
        if pb is None:
            continue
        box = PostprocessedBox(pb.left, pb.bot, pb.right, pb.top, pb.label, False)
        thick = int((h + w) // 300)
        if self.flags.output_type:
            resultsForJSON.append({"label": pb.label,
                                   "confidence": float('%.2f' % pb.max_prob),
                                   "topleft": {"x": pb.left, "y": pb.top},
                                   "bottomright": {"x": pb.right, "y": pb.bot}})
            writer.boxes.append(box)
            #continue

        mess = ' '.join([pb.label, str(round(pb.max_prob, 3))])
        cv2.rectangle(imgcv, (pb.left, pb.top), (pb.right, pb.bot), colors[pb.max_idx], 3)
        cv2.putText(imgcv, mess, (pb.left, pb.top - 12), 0, 1e-3 * h, self.meta['colors'][pb.max_idx], thick // 3)

    if not save:
        return imgcv

    img_name = os.path.join(self.flags.imgdir, os.path.basename(im))
    if "json" in self.flags.output_type:
        text_json = json.dumps(resultsForJSON)
        text_file = os.path.splitext(img_name)[0] + ".json"
        with open(text_file, 'w') as f:
            f.write(text_json)
    if "voc" in self.flags.output_type:
        img_file = os.path.splitext(img_name)[0] + XML_EXT
        writer.save(img_file)

# uncomment to write annotated images
# cv2.imwrite(img_name, imgcv)
