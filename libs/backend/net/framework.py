from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks.yolo import predict as predictv1
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2
from libs.io.flags import FlagIO
from os.path import basename


class Framework(FlagIO, object):
    constructor = vanilla.constructor
    loss = vanilla.trainer.loss

    def __init__(self, meta, flags):
        FlagIO.__init__(self, delay=0.5, subprogram=True)
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        self.meta = meta
        self.constructor(meta, flags)

    def is_inp(self, file_name):
        return True

    @classmethod
    def create(cls, meta, flags):
        net_type = meta['type']
        this = types.get(net_type, cls)
        return this(meta, flags)


class YOLO(Framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = predictv1.preprocess
    postprocess = predictv1.postprocess
    loss = yolo.train.loss
    is_inp = yolo.misc.is_inp
    profile = yolo.misc.profile
    # noinspection PyProtectedMember
    _batch = yolo.data._batch
    get_preprocessed_img = yolo.data.get_preprocessed_img
    get_feed_values = yolo.data.get_feed_values
    resize_input = predictv1.resize_input
    findboxes = predictv1.findboxes
    process_box = predictv1.process_box


class YOLOv2(Framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = predictv1.preprocess
    loss = yolov2.train.loss
    is_inp = yolo.misc.is_inp
    postprocess = predictv1.postprocess
    # noinspection PyProtectedMember
    _batch = yolov2.data._batch
    get_preprocessed_img = yolo.data.get_preprocessed_img
    get_feed_values = yolo.data.get_feed_values
    resize_input = predictv1.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = predictv1.process_box


class YOLOv3(Framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = predictv1.preprocess
    # loss = yolov3.train.loss  # TODO: yolov3.train
    is_inp = yolo.misc.is_inp
    postprocess = predictv1.postprocess
    # batch = yolov3.data._batch  # TODO: yolov3.data._batch
    resize_input = predictv1.resize_input
    # findboxes = yolov3.predict.findboxes  # TODO: yolov3.predict.findboxes
    process_box = predictv1.process_box

"""
framework factory
"""

types = {
    '[detection]': YOLO,
    '[region]': YOLOv2,
    '[yolo]': YOLOv3
}



