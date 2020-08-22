from __future__ import annotations
from typing import Generator, NamedTuple
from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2
from abc import abstractmethod
from libs.io.flags import FlagIO
from os.path import basename
from tensorflow import Tensor
import sys


class Framework(FlagIO, object):
    __create_key = object()

    def __init__(self, create_key, meta, flags):
        msg = "Framework must be created using Framework.create"
        if not create_key == Framework.__create_key:
            raise NotImplementedError(msg)
        FlagIO.__init__(self, delay=0.5, subprogram=True)
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        self.meta = meta
        self.flags = flags
        self.constructor(meta, flags)

    @classmethod
    @abstractmethod
    def register_type(cls) -> dict:
        """
        For registering Framework subclasses
        Returns:
            dict({meta['type']: cls})
        """
        return {'': cls}

    @abstractmethod
    def constructor(self, meta, flags) -> None:
        """Constructor to be defined in a framework module __init__.py"""
        pass

    @abstractmethod
    def parse(self, **kwargs) -> list:
        pass

    @abstractmethod
    def shuffle(self, parsed_data: list) -> Generator:
        pass

    @abstractmethod
    def loss(self, net_out: Tensor) -> None:
        """Setter method for self.loss"""
        pass

    @abstractmethod
    def is_input(self, filename) -> bool:
        """Checker for file format to be defined in framework module"""
        pass

    @classmethod
    def create(cls, meta, flags) -> Framework:
        """
        Factory method to create a Framework from Darknet configuration metadata and flags.
        """
        types = dict()
        for subclass in cls.__subclasses__():
            types.update(subclass.register_type())
        net_type = meta['type']
        this = types.get(net_type, cls)
        return this(cls.__create_key, meta, flags)


class Yolo(Framework):
    @classmethod
    def register_type(cls) -> dict:
        return {'[detection]': cls}

    constructor = yolo.constructor

    parse = yolo.data.parse
    shuffle = yolo.data.shuffle

    postprocess = yolo.predict.postprocess
    loss = yolo.train.loss
    is_input = yolo.misc.is_input

    batch = yolo.data.batch
    get_feed_values = yolo.data.get_feed_values
    get_preprocessed_img = yolo.data.get_preprocessed_img
    preprocess = yolo.predict.preprocess
    resize_input = yolo.predict.resize_input

    findboxes = yolo.predict.findboxes
    process_box = yolo.predict.process_box


class YoloV2(Framework):
    @classmethod
    def register_type(cls) -> dict:
        return {'[region]': cls}

    constructor = Yolo.constructor
    parse = Yolo.parse
    shuffle = Yolo.shuffle
    preprocess = Yolo.preprocess
    postprocess = Yolo.postprocess
    loss = yolov2.train.loss
    is_input = Yolo.is_input

    batch = yolov2.data.batch
    get_preprocessed_img = Yolo.get_preprocessed_img
    get_feed_values = Yolo.get_feed_values

    resize_input = Yolo.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = Yolo.process_box


class YoloV3(Framework):
    @classmethod
    def register_type(cls) -> dict:
        return {'[yolo]': cls}

    # Methods
    constructor = Yolo.constructor
    parse = Yolo.parse
    shuffle = yolov2.data.shuffle
    preprocess = Yolo.preprocess
    postprocess = Yolo.postprocess
    # loss = yolov3.train.loss  # TODO: yolov3.train.loss
    is_input = Yolo.is_input
    batch = YoloV2.batch
    get_feed_values = Yolo.get_feed_values
    resize_input = Yolo.resize_input
    # findboxes = yolov3.predict.findboxes  # TODO: yolov3.predict.findboxes
    process_box = Yolo.process_box


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
    loss = vanilla.train.loss
    constructor = vanilla.constructor

