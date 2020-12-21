import os
import cv2
from cv2 import rectangle, putText
import json
import numpy as np
from beagles.backend.net.frameworks.extensions.cy_yolo_findboxes import box_constructor
from beagles.io.pascalVoc import PascalVocWriter, XML_EXT
from beagles.base.box import PostprocessedBox, ProcessedBox


def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)

def resize_input(self, image):
    h, w, c = self.meta['inp_size']
    return cv2.resize(image, (w, h)) / 255.0 # casts from int8 to float32

def process(self, b, h, w, threshold):
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

def find(self, net_out):
    if type(net_out) is not np.ndarray:
        net_out = np.asarray(net_out)
    if net_out.ndim == 4:
        net_out = np.concatenate(net_out, 0)
    return box_constructor(self.meta, net_out, self.flags.threshold) or []

def preprocess(self, image, allobj = None):
    bboxes = None
    if type(image) is not np.ndarray:
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    if allobj is not None:  # in training mode
        try:
            image, bboxes = self.augment.spatial(image, allobj)
        except ValueError:
            self.logger.warn('Skipping augmentation of incompatible image')
        try:
            image = self.augment.pixel(image)
        except ValueError:
            self.logger.warn('Skipping augmentation of incompatible image')

    image = self.resize_input(image)

    if allobj is None:
        return image
    else:
        return image, bboxes  # , np.array(im) # for unit testing

def postprocess(self, net_out, image, save = True):
    if type(net_out) is not np.ndarray:
        net_out = np.asarray(net_out)

    boxes = self.find(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(image) is not np.ndarray:
        imgcv = cv2.imread(image)
    else:
        imgcv = image

    h, w, c = imgcv.shape
    writer = PascalVocWriter(self.flags.img_out, image, [h, w, c])
    resultsForJSON = []
    for b in boxes:
        pb = self.process(b, h, w, threshold)
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
        rectangle(imgcv, (pb.left, pb.top), (pb.right, pb.bot), colors[pb.max_idx], 3)
        putText(imgcv, mess, (pb.left, pb.top - 12), 0, 1e-3 * h, self.meta['colors'][pb.max_idx], thick // 3)

    if not save:
        return imgcv

    img_name = os.path.join(self.flags.imgdir, os.path.basename(image))
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
