from __future__ import annotations
import os
from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2
from libs.io.flags import FlagIO


class Framework(object):
    __create_key = object()
    token = dict()

    def __init__(self, create_key, *args):
        msg = f"Frameworks must be created using Framework.create"
        if not create_key == Framework.__create_key:
            raise NotImplementedError(msg)
        self.constructor(*args)

    @staticmethod
    def register_token(framework_token):
        """Decorator to register_token Framework subclass type tokens"""
        def deco(cls):
            types = framework_token.split(' ')
            multi = dict(zip(types, list([cls]) * len(types)))
            single = {framework_token: cls}
            cls.token = single if len(types) == 1 else multi
            return cls
        return deco

    @classmethod
    def create(cls, meta, flags) -> Framework:
        """
        Uses Darknet configuration metadata type token to find the right registered
        Framework subclass and passes metadata and flags into the subclass constructor method.
        """
        types = dict()
        for subclass in cls.__subclasses__():
            types.update(subclass.token)
        type_token = meta['type']
        this = types.get(type_token, None)
        if not this:
            raise KeyError(f'Unregistered framework type token: {type_token}')
        return this(cls.__create_key, meta, flags)

    def constructor(self, flags, meta):
        pass


@Framework.register_token('[detection]')
class Yolo(Framework):
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


@Framework.register_token('[region]')
class YoloV2(Framework):

    constructor = Yolo.constructor

    parse = Yolo.parse
    shuffle = Yolo.shuffle

    postprocess = Yolo.postprocess
    loss = yolov2.train.loss
    is_input = Yolo.is_input

    batch = yolov2.data.batch
    get_feed_values = Yolo.get_feed_values
    get_preprocessed_img = Yolo.get_preprocessed_img
    preprocess = Yolo.preprocess
    resize_input = Yolo.resize_input

    findboxes = yolov2.predict.findboxes
    process_box = Yolo.process_box


@Framework.register_token('[yolo]')
class YoloV3(Framework):

    # Methods
    constructor = Yolo.constructor

    parse = Yolo.parse
    shuffle = yolov2.data.shuffle

    postprocess = Yolo.postprocess
    # loss = yolov3.train.loss  # TODO: yolov3.train.loss
    is_input = Yolo.is_input

    batch = YoloV2.batch
    get_feed_values = Yolo.get_feed_values
    preprocess = Yolo.preprocess

    # findboxes = yolov3.predict.findboxes  # TODO: yolov3.predict.findboxes
    process_box = Yolo.process_box


@Framework.register_token('sse l1 l2 smooth sparse softmax')
class MultiLayerPerceptron(Framework):
    constructor = vanilla.constructor
    loss = vanilla.train.loss


