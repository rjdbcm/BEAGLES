from abc import abstractmethod
from beagles.base import SubsystemPrototype, Subsystem, register_subsystem
from beagles.backend.net.frameworks import vanilla
from beagles.backend.net.frameworks import yolo
from beagles.backend.net.frameworks import yolov2


class Framework(SubsystemPrototype):
    """
    SubsystemPrototype that uses Darknet configuration metadata type token to find a framework
    """
    def __init__(self, create_key, *args, **kwargs):
        super(Framework, self).__init__(create_key, *args, **kwargs)
        self.first = True

    @classmethod
    def create(cls, meta, flags):
        type_token = meta['type']
        types = cls.get_register()
        this = types.get(type_token, None)
        if not this:
            raise KeyError(f'Unregistered framework type token: {type_token}')
        return this(cls.create_key, meta, flags)

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def is_input(self, name):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def batch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def shuffle(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_feed_values(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_preprocessed(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def find(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process(self, *args, **kwargs):
        raise NotImplementedError


@register_subsystem(token='sse l1 l2 smooth sparse softmax', prototype=Framework)
class NeuralNet(Subsystem):
    constructor = vanilla.constructor
    loss = vanilla.train.loss


@register_subsystem(token='[detection]', prototype=Framework)
class Yolo(NeuralNet):
    constructor = yolo.constructor

    loss = yolo.train.loss

    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    is_input = yolo.data.is_input
    batch = yolo.data.batch
    get_feed_values = yolo.data.get_feed_values
    get_preprocessed = yolo.data.get_preprocessed

    find = yolo.predict.find
    resize_input = yolo.predict.resize_input
    preprocess = yolo.predict.preprocess
    process = yolo.predict.process
    postprocess = yolo.predict.postprocess


@register_subsystem(token='[region]', prototype=Framework)
class YoloV2(Yolo):
    constructor = Yolo.constructor

    loss = yolov2.train.loss

    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    is_input = yolo.data.is_input
    batch = yolov2.data.batch
    get_feed_values = yolo.data.get_feed_values
    get_preprocessed = yolo.data.get_preprocessed

    find = yolov2.predict.find
    resize_input = yolo.predict.resize_input
    preprocess = yolo.predict.preprocess
    process = yolo.predict.process
    postprocess = yolo.predict.postprocess


@register_subsystem(token='[yolo]', prototype=Framework)
class YoloV3(YoloV2):
    constructor = yolo.constructor

    # loss =

    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    is_input = yolo.data.is_input
    batch = yolov2.data.batch
    get_feed_values = yolo.data.get_feed_values
    get_preprocessed = yolo.data.get_preprocessed

    # find =
    resize_input = yolo.predict.resize_input
    # preprocess =
    # process =
    # postprocess =
