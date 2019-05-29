class Flags(dict):
    """Allows you to set dict values like attributes"""
    def __init__(self, defaults=True):
        if defaults:
            self.get_defaults()

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def get_defaults(self):
        self.train = False
        self.savepb = False
        self.demo = ''
        self.fbf = ''
        self.trainer = ''
        self.momentum = 0.0
        self.keep = 20
        self.batch = 16
        self.epoch = 64
        self.save = 16000
        self.lr = 1e-5
        self.clip = False
        self.saveVideo = './data/sample_img/out.avi'
        self.queue = 1
        self.lb = 0.0
        self.pbLoad = ''
        self.metaLoad = ''
        self.load = -1
        self.model = ''
        self.json = False
        self.gpu = 0.0
        self.gpuName = '/gpu:0'
        self.threshold = 0.1
        self.verbalise = True
        self.kill = False
        self.killed = False
        self.done = False
        self.progress = 0.0
        self.estimate = 0
        self.size = 0
        self.imgdir = './data/sample_img/'
        self.binary = './data/bin/'
        self.config = './data/cfg/'
        self.dataset = './data/committedframes/'
        self.backup = './data/ckpt/'
        self.labels = './data/predefined_classes.txt'
        self.log = './data/logs/flow.log'
        self.annotation = './data/committedframes/'
        self.summary = './data/summaries/'
