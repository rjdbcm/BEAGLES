from unittest import TestCase
from libs.dark.darknet import Darknet
from libs.utils.flags import Flags
from libs.dark.layer import Layer

meta = {'net': {'type': '[net]', 'batch': 1, 'subdivisions': 1, 'width': 608, 'height': 608,
             'channels': 3, 'momentum': 0.9, 'decay': 0.0005, 'angle': 0,
             'saturation': 1.5,
             'exposure': 1.5, 'hue': 0.1, 'learning_rate': 0.001, 'burn_in': 1000,
             'max_batches': 500200, 'policy': 'steps', 'steps': '400000,450000',
             'scales': '.1,.1'}, 'type': '[region]',
     'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
                 9.77052, 9.16828], 'bias_match': 1, 'classes': 80, 'coords': 4, 'num': 5,
     'softmax': 1, 'jitter': 0.3, 'rescore': 1, 'object_scale': 5, 'noobject_scale': 1,
     'class_scale': 1, 'coord_scale': 1, 'absolute': 1, 'thresh': 0.1, 'random': 1,
     'model': 'tests/yolov2.cfg', 'inp_size': [608, 608, 3], 'out_size': [19, 19, 425]}

layers = \
    [Layer('convolutional', 0, 3, 3, 32, 1, 1, 1, 'leaky'),
     Layer('maxpool', 1, 2, 2, 0),
     Layer('convolutional', 2, 3, 32, 64, 1, 1, 1, 'leaky'),
     Layer('maxpool', 3, 2, 2, 0),
     Layer('convolutional', 4, 3, 64, 128, 1, 1, 1, 'leaky'),
     Layer('convolutional', 5, 1, 128, 64, 1, 0, 1, 'leaky'),
     Layer('convolutional', 6, 3, 64, 128, 1, 1, 1, 'leaky'),
     Layer('maxpool', 7, 2, 2, 0),
     Layer('convolutional', 8, 3, 128, 256, 1, 1, 1, 'leaky'),
     Layer('convolutional', 9, 1, 256, 128, 1, 0, 1, 'leaky'),
     Layer('convolutional', 10, 3, 128, 256, 1, 1, 1, 'leaky'),
     Layer('maxpool', 11, 2, 2, 0),
     Layer('convolutional', 12, 3, 256, 512, 1, 1, 1, 'leaky'),
     Layer('convolutional', 13, 1, 512, 256, 1, 0, 1, 'leaky'),
     Layer('convolutional', 14, 3, 256, 512, 1, 1, 1, 'leaky'),
     Layer('convolutional', 15, 1, 512, 256, 1, 0, 1, 'leaky'),
     Layer('convolutional', 16, 3, 256, 512, 1, 1, 1, 'leaky'),
     Layer('maxpool', 17, 2, 2, 0),
     Layer('convolutional', 18, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 19, 1, 1024, 512, 1, 0, 1, 'leaky'),
     Layer('convolutional', 20, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 21, 1, 1024, 512, 1, 0, 1, 'leaky'),
     Layer('convolutional', 22, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 23, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 24, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('route', 25, [16]),
     Layer('convolutional', 26, 1, 512, 64, 1, 0, 1, 'leaky'),
     Layer('reorg', 27, 2),
     Layer('route', 28, [27, 24]),
     Layer('convolutional', 29, 3, 1280, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 30, 1, 1024, 425, 1, 0, False, 'linear')]


class TestDarknet(TestCase):
    def testParseAndYieldConfig(self):
        self.maxDiff = None
        flags = Flags()
        flags.model = 'tests/yolov2.cfg'
        darknet = Darknet(flags)
        self.assertDictEqual(darknet.meta, meta, 'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, layers)