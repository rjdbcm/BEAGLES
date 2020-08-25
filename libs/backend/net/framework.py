from __future__ import annotations
from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2
from abc import abstractmethod
from libs.io.flags import FlagIO
from os.path import basename


class FrameworkMeta(type):
    def __new__(mcs, name, bases, dct):
        if "__init__" in dct:
            raise NameError("Framework subclasses should not have an __init__ method")
        return type.__new__(mcs, name, bases, dct)


class Framework(FlagIO, object):
    __metaclass__ = FrameworkMeta
    __create_key = object()
    token = dict()

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

    @staticmethod
    def register(framework_token):
        """Decorator to register Framework subclass type tokens"""
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
        Factory method to create a Framework.
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

    @abstractmethod
    def constructor(self, meta, flags) -> str:
        """Constructor to be defined in a framework module __init__.py"""
        pass

    @abstractmethod
    def loss(self):
        pass


@Framework.register('[detection]')
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


@Framework.register('[region]')
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


@Framework.register('[yolo]')
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


@Framework.register('sse l1 l2 smooth sparse softmax')
class MultiLayerPerceptron(Framework):
    constructor = vanilla.constructor
    loss = vanilla.train.loss


