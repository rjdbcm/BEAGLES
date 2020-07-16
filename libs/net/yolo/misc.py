import pickle
import numpy as np
# import filetype
import cv2
import os


def labels(meta, flags):
    file = flags.labels
    model = os.path.basename(meta['name'])
    with open(file, 'r') as f:
        meta['labels'] = list()
        labs = [l.strip() for l in f.readlines()]
        for lab in labs:
            if lab.startswith("#"):
                continue
            meta['labels'] += [lab]


def is_inp(self, name):
    """checks if input has a valid image file extension"""
    # TODO: Replace with a filetype based checker
    return name.lower().endswith(('.jpg', '.jpeg', '.png'))


def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] // S
        cx = a + obj[1]
        cy = b + obj[2]
        centerx = cx * cellx
        centery = cy * celly
        ww = obj[3]**2 * w
        hh = obj[4]**2 * h
        cv2.rectangle(im,
                      (int(centerx - ww/2), int(centery - hh/2)),
                      (int(centerx + ww/2), int(centery + hh/2)),
                      (0, 0, 255), 2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()


def profile(self):
    pass
