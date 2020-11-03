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
        'clip_norm': (0.0,                        float, 'Gradient Clip Norm'),
        'clr_mode': ('triangular2',                 str, 'Cyclic Learning Policy'),
        'done': (False,                            bool, 'Done Signal'),
        'epoch': (1,                                int, 'Epochs to Train'),
        'error': ('',                               str, 'Error Signal'),
        'video': ([],                              list, 'Videos to Annotate'),
        'gpu': (0.0,                              float, 'GPU Utilization'),
        'gpu_name': ('/gpu:0',                      str, 'Current GPU'),
        'output_type': ([],                        list, 'Predict Output Type'),
        'keep': (20,                                int, 'Checkpoint to Keep'),
        'kill': (False,                            bool, 'Kill Signal'),
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

def get_defaults(k):
    data = DEFAULTS['FLAGS'].get(k)
    dtype = DEFAULTS['TYPES'].get(k)
    desc = DEFAULTS['DESCS'].get(k)
    return k, data, dtype, desc

def gen_defaults():
    for flag in _FLAGS:
        yield get_defaults(flag)

class Flags(dict):
    """
    Allows you to set and get {key: value} pairs like attributes.
    Enforces type-checking during flag setting.
    This allows compatibility with argparse.Namespace objects.
    """

    def __init__(self, defaults=True):
        super(Flags, self).__init__()
        if defaults:
            self._get_defaults()

    def _get_defaults(self):
        for flag, value, *_ in gen_defaults():
            self.__setattr__(flag, value)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        attr, _, dtype, _ = get_defaults(attr)
        if isinstance(value, dtype):
            self[attr] = value
        else:
            raise ValueError(f'Expected type {dtype} for {attr} found {type(value)}')

    def from_json(self, file):
        data = dict(json.load(file))
        for attr, value in data.items():
            self.__setattr__(attr, value)
        return self

    def to_json(self, file=sys.stdout):
        return json.dump(dict(self.items()), fp=file)
