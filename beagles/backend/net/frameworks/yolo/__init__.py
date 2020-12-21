from beagles.backend.net.frameworks.yolo import data, train, predict
from beagles.backend.net.augment import Augment
from beagles.io.logs import get_logger
import numpy as np
import os
import sys

""" YOLO framework __init__ equivalent"""


def constructor(self, meta, flags):

    def to_color(idx, b):
        """Returns (blu, red, grn) tuple"""
        b2 = b * b
        blu = 2 - idx / b2
        red = 2 - (idx % b2) / b
        grn = 2 - (idx % b2) % b
        return blu * 127, red * 127, grn * 127

    self.flags = flags
    self.logger, self.logfile = get_logger()
    model = os.path.basename(meta['model'])
    model = '.'.join(model.split('.')[:-1])
    meta['name'] = model

    try:
        meta['soft_nms'] = meta['soft_nms']
    except KeyError:
        meta['soft_nms'] = 0

    tx_args = []
    try: # look for optional darknet config file albumentations extensions
        tx_args = meta['net']['augment']
    except KeyError:
        pass
    self.augment = Augment(*tx_args)
    if tx_args:
        for line in repr(self.augment).splitlines():
            self.logger.info(line)

    if 'labels' not in meta:
        with open(flags.labels, 'r') as f:
            meta['labels'] = list()
            labs = [l.strip() for l in f.readlines()]
            for lab in labs:
                if lab.startswith("#"):
                    continue
                meta['labels'] += [lab]
    # We're not loading from a .pb so we do need to load the labels
    try:
        assert len(meta['labels']) == meta['classes'], \
            f'{self.flags.labels} and {meta["model"]} indicate inconsistent class numbers'
    except AssertionError as e:
        self.flags.error = str(e)
        self.logger.error(str(e))
        self.flags.kill = True
        self.io.send_flags()
        raise

    # assign a color for each label
    base = int(np.ceil(pow(meta['classes'], 1. / 3)))
    meta['colors'] = [to_color(x, base) for x in range(len(meta['labels']))]
    self.fetch = list()
    self.meta = meta

    # over-ride the threshold in meta if flags has it.
    if self.flags.threshold > 0.0:
        self.meta['thresh'] = self.flags.threshold
