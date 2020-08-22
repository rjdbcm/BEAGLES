from __future__ import annotations
from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2
from abc import abstractmethod
from libs.io.flags import FlagIO
from os.path import basename
import sys


class Framework(FlagIO, object):
    constructor = vanilla.constructor

    def __init__(self, meta, flags):
        FlagIO.__init__(self, delay=0.5, subprogram=True)
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        self.meta = meta
        # noinspection PyArgumentList
        self.constructor(meta, flags)

    @staticmethod
    @abstractmethod
    def register_type() -> dict:
        return {str(): Framework}

    @abstractmethod
    def parse(self, **kwargs):
        pass

    @abstractmethod
    def loss(self, net_out):
        pass

    @abstractmethod
    def is_inp(self, filename):
        pass

    @classmethod
    def create(cls, meta, flags) -> Framework:
        types = dict()
        for subclass in cls.__subclasses__():
            types.update(subclass.register_type())
        net_type = meta['type']
        this = types.get(net_type, cls)
        return this(meta, flags)


class YOLO(Framework):
    @staticmethod
    def register_type() -> dict:
        return {'[detection]': YOLO}

    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = yolo.predict.preprocess
    postprocess = yolo.predict.postprocess
    loss = yolo.train.loss
    is_inp = yolo.misc.is_inp
    profile = yolo.misc.profile
    batch = yolo.data.batch
    get_preprocessed_img = yolo.data.get_preprocessed_img
    get_feed_values = yolo.data.get_feed_values
    resize_input = yolo.predict.resize_input
    findboxes = yolo.predict.findboxes
    process_box = yolo.predict.process_box


class YOLOv2(Framework):
    @staticmethod
    def register_type() -> dict:
        return {'[region]': YOLOv2}

    # Methods
    constructor = YOLO.constructor
    parse = YOLO.parse
    shuffle = YOLO.shuffle
    preprocess = YOLO.preprocess
    loss = yolov2.train.loss
    is_inp = YOLO.is_inp
    postprocess = YOLO.postprocess
    batch = yolov2.data.batch
    get_preprocessed_img = YOLO.get_preprocessed_img
    get_feed_values = YOLO.get_feed_values
    resize_input = YOLO.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = YOLO.process_box


class YOLOv3(Framework):
    @staticmethod
    def register_type() -> dict:
        return {'[yolo]': YOLOv3}

    # Methods
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = yolo.predict.preprocess
    # loss = yolov3.train.loss  # TODO: yolov3.train
    is_inp = yolo.misc.is_inp
    postprocess = yolo.predict.postprocess
    # batch = yolov3.data.batch  # TODO: yolov3.data.batch
    resize_input = yolo.predict.resize_input
    # findboxes = yolov3.predict.findboxes  # TODO: yolov3.predict.findboxes
    process_box = yolo.predict.process_box


class Vanilla(Framework):
    def type(self) -> dict:
        types = {
            'sse': Vanilla,
            'l2': Vanilla,
            'sparse': Vanilla,
            'l1': Vanilla,
            'softmax': Vanilla,
        }
        return types
    loss = vanilla.trainer.loss
