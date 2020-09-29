import sys
import json


_FLAGS = {
        'annotation': ('./data/committedframes/',   str, 'Image Annotations Path'),
        'dataset': ('./data/committedframes/',      str, 'Images Path'),
        'backup': ('./data/ckpt/',                  str, 'Checkpoints Path'),
        'summary': ('./data/summaries/',            str, 'Tensorboard Summaries Path'),
        'log': ('./data/logs/flow.log',             str, 'Log File Path'),
        'config': ('./data/cfg/',                   str, 'Model Config Path'),
        'binary': ('./data/bin/',                   str, 'Binary Weights Path'),
        'built_graph': ('./data/built_graph/',      str, 'Protobuf Output Path'),
        'imgdir': ('./data/sample_img/',            str, 'Images to Predict Path'),
        'img_out': ('./data/img_out/',              str, 'Prediction Output Path'),
        'video_out': ('./data/video_out/',          str, 'Video Output Path'),
        'batch': (16,                               int, 'Images per Batch'),
        'cli': (False,                             bool, 'Using Command Line'),
        'clip': (False,                            bool, 'Clipping Gradients'),
        'clip_norm': (0,                          float, 'Gradient Clip Norm'),
        'clr_mode': ('triangular2',                 str, 'Cyclic Learning Policy'),
        'done': (False,                            bool, 'Done Signal'),
        'epoch': (1,                                int, 'Epochs to Train'),
        'error': ('',                               str, 'Error Signal'),
        'video': ([],                              list, 'Videos to Annotate'),
        'gpu': (0.0,                              float, 'GPU Utilization'),
        'gpu_name': ('/gpu:0',                      str, 'Current GPU'),
        'output_type': ([],                         str, 'Predict Output Type'),
        'keep': (20,                                int, 'Checkpoint to Keep'),
        'kill': (False,                             str, 'Kill Signal'),
        'labels': ('./data/predefined_classes.txt', str, 'Class Labels File'),
        'load': (-1,                                int, 'Checkpoint to Use'),
        'lr': (1e-05,                             float, 'Initial Learning Rate'),
        'max_lr': (1e-05,                         float, 'Maximum Learning Rate'),
        'model': ('',                               str, 'Model Configuration File'),
        'momentum': (0.0,                         float, 'Momentum Setting for Trainer'),
        'progress': (0.0,                         float, 'Progress Signal'),
        'project_name': ('default',                 str, 'Saving Under'),
        'save': (16000,                             int, 'Save Checkpoint After'),
        'size': (1,                                 int, 'Dataset Size (Images)'),
        'started': (False,                          int, 'Started Signal'),
        'step_size_coefficient': (2,                int, 'Cyclic Learning Coefficient'),
        'threshold': (0.4,                        float, 'Detection Record Threshold'),
        'trainer': ('rmsprop',                      str, 'Optimization Algorithm'),
        'verbalise': (False,                       bool, 'Verbose Output'),
        'train': (False,                           bool, 'Training Mode')
        }


DEFAULTS = {
   'FLAGS': {k: v[0] for k, v in _FLAGS.items()},
   'TYPES': {k: v[1] for k, v in _FLAGS.items()},
   'DESCS': {k: v[2] for k, v in _FLAGS.items()}
}


class Flags(dict):
    """
    Allows you to set and get {key: value} pairs like attributes.
    This allows compatibility with argparse.Namespace objects.
    """

    def __init__(self, defaults=True):
        super(Flags, self).__init__()
        if defaults:
            self.get_defaults()

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    # noinspection PyMethodMayBeStatic
    def get_defaults(self):
        self.annotation = './data/committedframes/'
        self.backup = './data/ckpt/'
        self.batch = 16
        self.binary = './data/bin/'
        self.built_graph = './data/built_graph/'
        self.cli = False
        self.clip = False
        self.clip_norm = 0
        self.clr_mode = 'triangular2'
        self.config = './data/cfg/'
        self.dataset = './data/committedframes/'
        self.done = False
        self.epoch = 1
        self.error = ''
        self.video = []
        self.gpu = 0.0
        self.gpu_name = '/gpu:0'
        self.imgdir = './data/sample_img/'
        self.img_out = './data/img_out/'
        self.output_type = []
        self.keep = 20
        self.kill = False
        self.labels = './data/predefined_classes.txt'
        self.load = -1
        self.log = './data/logs/flow.log'
        self.lr = 1e-05
        self.max_lr = 1e-05
        self.model = ''
        self.momentum = 0.0
        self.progress = 0.0
        self.project_name = 'default'
        self.save = 16000
        self.save_video = True
        self.size = 1
        self.started = False
        self.step_size_coefficient = 2
        self.summary = './data/summaries/'
        self.threshold = 0.4
        self.timeout = 0
        self.trainer = 'rmsprop'
        self.verbalise = False
        self.video_out = './data/video_out/'
        self.train = False

    def from_json(self, file):
        data = dict(json.load(file))
        for attr, value in data.items():
            self.__setattr__(attr, value)
        return self

    def to_json(self, file=sys.stdout):
        return json.dump(dict(self.items()), fp=file)
