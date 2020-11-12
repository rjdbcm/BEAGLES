from unittest import TestCase
from beagles.backend.darknet import Darknet
from beagles.backend.net.framework import Framework
from beagles.backend.io.darknet_config_file import DarknetConfigFile
from beagles.base.errors import DarknetConfigEmpty
from beagles.base.flags import Flags
from beagles.backend.darknet.layer import Layer

meta = {
    'net': {'type': '[net]', 'batch': 1, 'subdivisions': 1, 'width': 608, 'height': 608,
            'channels': 3, 'momentum': 0.9, 'decay': 0.0005, 'angle': 0,
            'saturation': 1.5,
            'exposure': 1.5, 'hue': 0.1, 'learning_rate': 0.001, 'burn_in': 1000,
            'max_batches': 500200, 'policy': 'steps', 'steps': '400000,450000',
            'scales': '.1,.1'}, 'type': '[region]',
    'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
                9.77052, 9.16828], 'bias_match': 1, 'classes': 4,
    'coords': 4, 'num': 5,
    'softmax': 1, 'jitter': 0.3, 'rescore': 1, 'object_scale': 5, 'noobject_scale': 1,
    'class_scale': 1, 'coord_scale': 1, 'absolute': 1, 'thresh': 0.1, 'random': 1,
    'model': 'tests/resources/test.cfg', 'inp_size': [608, 608, 3], 'out_size': 16245}

yolov1_meta = {
    'net': {'type': '[net]', 'batch': 64, 'subdivisions': 64, 'height': 448, 'width': 448,
            'channels': 3, 'momentum': 0.9, 'decay': 0.0005, 'learning_rate': 0.0001,
            'policy': 'steps', 'steps': '20,40,60,80,20000,30000',
            'scales': '5,5,2,2,.1,.1', 'max_batches': 40000}, 'type': '[detection]',
    'classes': 4, 'coords': 4, 'rescore': 1, 'side': 7, 'num': 2, 'softmax': 0, 'sqrt': 1,
    'jitter': 0.2, 'object_scale': 1, 'noobject_scale': 0.5, 'class_scale': 1,
    'coord_scale': 5, 'model': 'tests/resources/test_yolov1.cfg',
    'inp_size': [448, 448, 3], 'out_size': 686}

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
     Layer('avgpool', 14),
     Layer('convolutional', 15, 3, 256, 512, 1, 1, 1, 'leaky'),
     Layer('convolutional', 16, 1, 512, 256, 1, 0, 1, 'leaky'),
     Layer('convolutional', 17, 3, 256, 512, 1, 1, 1, 'leaky'),
     Layer('maxpool', 18, 2, 2, 0),
     Layer('convolutional', 19, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 20, 1, 1024, 512, 1, 0, 1, 'leaky'),
     Layer('convolutional', 21, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 22, 1, 1024, 512, 1, 0, 1, 'leaky'),
     Layer('convolutional', 23, 3, 512, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 24, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('convolutional', 25, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('route', 26, [17]),
     Layer('convolutional', 27, 1, 512, 64, 1, 0, 1, 'leaky'),
     Layer('reorg', 28, 2),
     Layer('route', 29, [28, 25]),
     Layer('convolutional', 30, 3, 1280, 1024, 1, 1, 1, 'leaky'),
     Layer('lstm', 31, 1024, 1),
     Layer('connected', 32, 369664, 4096, 'ramp'),
     Layer('dropout', 33, 0.5),
     Layer('connected', 34, 4096, 1000, 'ramp'),
     Layer('convolutional', 35, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('shortcut', 36, 32),
     Layer('convolutional', 37, 1, 1024, 45, 1, 0, False, 'linear')]

yolov1_layer = [Layer('crop', 0),
                Layer('convolutional', 1, 3, 3, 16, 1, 1, False, 'leaky'),
                Layer('maxpool', 2, 2, 2, 0),
                Layer('convolutional', 3, 3, 16, 32, 1, 1, False, 'leaky'),
                Layer('maxpool', 4, 2, 2, 0),
                Layer('convolutional', 5, 3, 32, 64, 1, 1, False, 'leaky'),
                Layer('maxpool', 6, 2, 2, 0),
                Layer('convolutional', 7, 3, 64, 128, 1, 1, False, 'leaky'),
                Layer('maxpool', 8, 2, 2, 0),
                Layer('convolutional', 9, 3, 128, 256, 1, 1, False, 'leaky'),
                Layer('maxpool', 10, 2, 2, 0),
                Layer('convolutional', 11, 3, 256, 512, 1, 1, False, 'leaky'),
                Layer('maxpool', 12, 2, 2, 0),
                Layer('convolutional', 13, 3, 512, 1024, 1, 1, False, 'leaky'),
                Layer('convolutional', 14, 3, 1024, 1024, 1, 1, False, 'leaky'),
                Layer('convolutional', 15, 3, 1024, 1024, 1, 1, False, 'leaky'),
                Layer('flatten', 16),
                Layer('connected', 17, 256, 4096, 'leaky'),
                Layer('dropout', 18, 0.5),
                Layer('connected', 19, 4096, 1470, 'linear')]


class TestDarknet(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flags = Flags()
        cls.maxDiff = None

    def testFrameworkImproperInitialization(self):
        darknet = Darknet(self.flags)
        self.assertRaises(NotImplementedError, Framework, self, darknet.meta, self.flags)

    def testUnregisteredFrameworkToken(self):
        self.flags.model = 'tests/resources/test_phony.cfg'
        darknet = Darknet(self.flags)
        self.assertRaises(KeyError, Framework.create, darknet.meta, self.flags)

    def testParseAndYieldYoloV2Config(self):
        self.flags.model = 'tests/resources/test.cfg'
        self.flags.labels = 'tests/resources/test_classes.txt'
        darknet = Darknet(self.flags)
        self.assertDictEqual(darknet.meta, meta,
                             'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, layers)

    def testParseAndYieldYoloV1Config(self):
        self.flags.labels = 'tests/resources/test_classes.txt'
        self.flags.model = 'tests/resources/test_yolov1.cfg'
        darknet = Darknet(self.flags)
        self.assertDictEqual(darknet.meta, yolov1_meta,
                             'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, yolov1_layer)

    def testDarknetConfigToAndFromJson(self):
        self.flags.model = 'tests/resources/test.cfg'
        cfg = DarknetConfigFile(self.flags.model)
        json_cfg_file = cfg.to_json()
        json_cfg = DarknetConfigFile(json_cfg_file)
        self.assertEqual(cfg, json_cfg, 'layers mismatch')
        self.assertRaises(FileNotFoundError, DarknetConfigFile,
                          'tests/resources/phonybologna.cfg')
        self.flags.model = json_cfg_file
        self.flags.labels = 'tests/resources/test_classes.txt'
        darknet = Darknet(self.flags)
        self.assertDictEqual(darknet.meta, meta,
                             'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, layers)

    def testEmptyDarknetConfigFile(self):
        self.assertRaises(DarknetConfigEmpty, DarknetConfigFile,
                          'tests/resources/empty.cfg')
