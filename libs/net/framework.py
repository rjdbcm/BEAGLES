from . import yolo
from . import yolov2
from . import vanilla
from ..utils.flags import FlagIO
from os.path import basename


class framework(FlagIO, object):
    constructor = vanilla.constructor
    loss = vanilla.train.loss
    
    def __init__(self, meta, flags):
        FlagIO.__init__(self, delay=0.5, subprogram=True)
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        self.meta = meta
        self.constructor(meta, flags)

    def is_inp(self, file_name):
        return True


class YOLO(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = yolo.predict.preprocess
    postprocess = yolo.predict.postprocess
    loss = yolo.train.loss
    is_inp = yolo.misc.is_inp
    profile = yolo.misc.profile
    # noinspection PyProtectedMember
    _batch = yolo.data._batch
    resize_input = yolo.predict.resize_input
    findboxes = yolo.predict.findboxes
    process_box = yolo.predict.process_box


class YOLOv2(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = yolo.predict.preprocess
    loss = yolov2.train.loss
    is_inp = yolo.misc.is_inp
    postprocess = yolo.predict.postprocess
    # noinspection PyProtectedMember
    _batch = yolov2.data._batch
    resize_input = yolo.predict.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = yolo.predict.process_box


class YOLOv3(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = yolo.predict.preprocess
    # loss = yolov3.train.loss  # TODO: yolov3.train
    is_inp = yolo.misc.is_inp
    postprocess = yolo.predict.postprocess
    # batch = yolov3.data._batch  # TODO: yolov3.data._batch
    resize_input = yolo.predict.resize_input
    # findboxes = yolov3.predict.findboxes  # TODO: yolov3.predict.findboxes
    process_box = yolo.predict.process_box

"""
framework factory
"""

types = {
    '[detection]': YOLO,
    '[region]': YOLOv2,
    '[yolo]': YOLOv3
}


def create_framework(meta, flags):
    net_type = meta['type']
    this = types.get(net_type, framework)
    return this(meta, flags)
