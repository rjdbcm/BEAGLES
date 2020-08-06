from unittest import TestCase
from libs.net.framework import Framework
from libs.dark.darknet import Darknet
from libs.utils.darknet_config_file import DarknetConfigFile
from libs.utils.errors import DarknetConfigEmpty
from libs.utils.flags import Flags
from libs.dark.layer import Layer

meta = {
    'net': {'type': '[net]', 'batch': 1, 'subdivisions': 1, 'width': 608, 'height': 608,
            'channels': 3, 'momentum': 0.9, 'decay': 0.0005, 'angle': 0,
            'saturation': 1.5,
            'exposure': 1.5, 'hue': 0.1, 'learning_rate': 0.001, 'burn_in': 1000,
            'max_batches': 500200, 'policy': 'steps', 'steps': '400000,450000',
            'scales': '.1,.1'}, 'type': '[region]',
    'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
                9.77052, 9.16828], 'bias_match': 1, 'classes': 4,
    'colors': [(254.0, 254.0, 254),
               (222.25, 190.5, 127),
               (190.5, 127.0, 254),
               (158.75, 63.5, 127)],
    'coords': 4, 'num': 5,
    'softmax': 1, 'jitter': 0.3, 'rescore': 1, 'object_scale': 5, 'noobject_scale': 1,
    'class_scale': 1, 'coord_scale': 1, 'absolute': 1, 'thresh': 0.4, 'random': 1,
    'labels': ['foo', 'bar', 'baz', 'bah'],
    'name': 'test',
    'model': 'tests/resources/test.cfg', 'inp_size': [608, 608, 3], 'out_size': 16245}

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
     Layer('connected', 31, 369664, 4096, 'ramp'),
     Layer('dropout', 32, 0.5),
     Layer('connected', 33, 4096, 1000, 'ramp'),
     Layer('convolutional', 34, 3, 1024, 1024, 1, 1, 1, 'leaky'),
     Layer('select', 35, 369664, 1470, 'linear', None, 686, [8, 14, 15, 19, 28, 34, 35, 39, 48, 54, 55, 59, 68, 74, 75, 79, 88, 94, 95, 99, 108, 114, 115, 119, 128, 134, 135, 139, 148, 154, 155, 159, 168, 174, 175, 179, 188, 194, 195, 199, 208, 214, 215, 219, 228, 234, 235, 239, 248, 254, 255, 259, 268, 274, 275, 279, 288, 294, 295, 299, 308, 314, 315, 319, 328, 334, 335, 339, 348, 354, 355, 359, 368, 374, 375, 379, 388, 394, 395, 399, 408, 414, 415, 419, 428, 434, 435, 439, 448, 454, 455, 459, 468, 474, 475, 479, 488, 494, 495, 499, 508, 514, 515, 519, 528, 534, 535, 539, 548, 554, 555, 559, 568, 574, 575, 579, 588, 594, 595, 599, 608, 614, 615, 619, 628, 634, 635, 639, 648, 654, 655, 659, 668, 674, 675, 679, 688, 694, 695, 699, 708, 714, 715, 719, 728, 734, 735, 739, 748, 754, 755, 759, 768, 774, 775, 779, 788, 794, 795, 799, 808, 814, 815, 819, 828, 834, 835, 839, 848, 854, 855, 859, 868, 874, 875, 879, 888, 894, 895, 899, 908, 914, 915, 919, 928, 934, 935, 939, 948, 954, 955, 959, 968, 974, 975, 979], 980),
     Layer('shortcut', 36, {'type': '[convolutional]', 'batch_normalize': 1, 'size': 3,
                            'stride': 1, 'pad': 1, 'filters': 1024, 'activation': 'leaky',
                            '_size': [19, 19, 1024, 369664, True]}),
     Layer('convolutional', 37, 1, 1024, 45, 1, 0, False, 'linear')]


class TestDarknet(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flags = Flags()
        cls.maxDiff = None
        cls.flags.model = 'tests/resources/test.cfg'

    def testParseAndYieldYoloV2Config(self):
        self.flags.labels = 'tests/resources/test_classes.txt'
        darknet = Darknet(self.flags)
        Framework.create(darknet.meta, self.flags)
        self.assertDictEqual(darknet.meta, meta,
                             'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, layers)

    def testDarknetConfigToAndFromJson(self):
        cfg = DarknetConfigFile(self.flags.model)
        json_cfg_file = cfg.to_json()
        json_cfg = DarknetConfigFile(json_cfg_file)
        self.assertEqual(cfg, json_cfg, 'layers mismatch')
        self.assertRaises(FileNotFoundError, DarknetConfigFile, 'tests/resources/phonybologna.cfg')
        self.flags.model = json_cfg_file
        self.flags.labels = 'tests/resources/test_classes.txt'
        darknet = Darknet(self.flags)
        Framework.create(darknet.meta, self.flags)
        self.assertDictEqual(darknet.meta, meta,
                             'Failed to correctly parse darknet metadata')
        self.assertEqual(darknet.layers, layers)

    def testEmptyDarknetConfigFile(self):
        self.assertRaises(DarknetConfigEmpty, DarknetConfigFile, 'tests/resources/empty.cfg')