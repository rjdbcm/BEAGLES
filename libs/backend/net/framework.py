from libs.backend.base import SubsystemPrototype, Subsystem, register_subsystem
from libs.backend.net.frameworks import vanilla
from libs.backend.net.frameworks import yolo
from libs.backend.net.frameworks import yolov2


class Framework(SubsystemPrototype):

    def __init__(self, create_key, *args, **kwargs):
        super(Framework, self).__init__(create_key, *args, **kwargs)

    @classmethod
    def create(cls, meta, flags):
        """
        Uses Darknet configuration metadata type token to find the right registered
        Subsystem and passes metadata and flags into the Subsystem's constructor method.
        """
        type_token = meta['type']
        types = cls.get_register()
        this = types.get(type_token, None)
        if not this:
            raise KeyError(f'Unregistered framework type token: {type_token}')
        return this(cls.create_key, meta, flags)


@register_subsystem(token='[detection]', prototype=Framework)
class Yolo(Subsystem):
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


@register_subsystem(token='[region]', prototype=Framework)
class YoloV2(Yolo):
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


@register_subsystem(token='sse l1 l2 smooth sparse softmax', prototype=Framework)
class MultiLayerPerceptron(Subsystem):
    constructor = vanilla.constructor
    loss = vanilla.train.loss


