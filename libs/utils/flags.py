

class Flags(dict):
    """
    Allows you to set and get {key: value} pairs like attributes.
    This allows compatibility with argparse.Namespace objects.
    """

    def __init__(self, defaults=True):
        if defaults:
            # All paths are relative to BEAGLES.py
            self.annotation = './data/committedframes/'
            self.backup = './data/ckpt/'
            self.batch = 16
            self.binary = './data/bin/'
            self.built_graph = './data/built_graph/'
            self.cli = False
            self.clip = False
            self.clip_norm = 5
            self.clr_mode = "triangular2"
            self.config = './data/cfg/'
            self.dataset = './data/committedframes/'
            self.demo = ''
            self.done = False
            self.epoch = 1
            self.error = ""
            self.fbf = ''
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
            self.lr = 1.0e-5
            self.max_lr = 1.0e-5
            self.model = ''
            self.momentum = 0.0
            self.progress = 0.0
            self.project_name = "default"
            self.save = 16000
            self.freeze = False
            self.save_video = True
            self.size = 0
            self.started = False
            self.step_size_coefficient = 2
            self.summary = './data/summaries/'
            self.threshold = 0.4
            self.timeout = 0
            self.trainer = 'rmsprop'
            self.verbalise = False
            self.video_out = "./data/video_out/"
            self.train = False
            self.pb_load = False
            self.meta_load = False

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        pass

    def get_defaults(self):
        self.__init__()
