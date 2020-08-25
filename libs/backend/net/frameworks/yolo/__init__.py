from libs.backend.net.frameworks.yolo import data, misc, train, predict
from libs.io.flags import FlagIO
import numpy as np

""" YOLO framework __init__ equivalent"""


def constructor(self, meta, flags):
    self.flags = flags
    self.logger.info(f'Framework type: {self.token}')

    def _to_color(indx, base):
        """ return (b, r, g) tuple"""
        base2 = base * base
        b = 2 - indx / base2
        r = 2 - (indx % base2) / base
        g = 2 - (indx % base2) % base
        return (b * 127, r * 127, g * 127)

    if 'labels' not in meta:
        misc.labels(meta, flags)  # We're not loading from a .pb so we do need to load the labels
    try:
        assert len(meta['labels']) == meta['classes'], (
                '{} and {} indicate inconsistent class numbers').format(flags.labels, meta['model'])
    except AssertionError as e:
        self.flags.error = str(e)
        self.logger.error(str(e))
        FlagIO.send_flags(self)
        raise


    # assign a color for each label
    colors = list()
    base = int(np.ceil(pow(meta['classes'], 1. / 3)))
    for x in range(len(meta['labels'])):
        colors += [_to_color(x, base)]
    meta['colors'] = colors
    self.fetch = list()
    self.meta, self.flags = meta, flags

    # over-ride the threshold in meta if flags has it.
    if flags.threshold > 0.0:
        self.meta['thresh'] = flags.threshold
