from collections import namedtuple


class BoundingBox(object):
    def __init__(self):
        self.boxlist = []

    def addBndBox(self, bounding_box: namedtuple, metadata):
        bndbox = namedtuple('bndbox', 'xmin ymin xmax ymax label difficult')
        bndbox = bndbox(bounding_box.xmin, bounding_box.ymin,
                        bounding_box.xmax, bounding_box.ymax,
                        metadata.label, metadata.difficult)
        self.boxlist.append(bndbox)