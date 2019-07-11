from datetime import datetime
from libs.net.help import say
import pickle
import sys
import time
import os


class FlagIO(object):
    def __init__(self, subprogram=False, delay=0.1):
        self.READ_MSG = "[{}] {} Flags Read: {}"
        self.subprogram = subprogram
        self.delay = delay
        self.flagpath = self.init_ramdisk()
        if subprogram:
            try:
                f = open(self.flagpath)
                f.close()
            except FileNotFoundError:
                time.sleep(1)

    def debug_log(self, *msgs):
        with open(self.flags.log, 'a') as logfile:
            msg = list(msgs)
            form = "[{}] {}\n"
            for msg in msgs:
                if msg is None:
                    continue
                else:
                    logfile.write(form.format(datetime.now(), msg))
        logfile.close()


    def send_flags(self):
        self.debug_log("{} Flags Send: {}".format(type(self).__name__, self.flags))
        with open(r"{}".format(self.flagpath), "wb") as outfile:
            pickle.dump(self.flags, outfile)

    def read_flags(self):
        inpfile = None
        count = 0
        while inpfile is None:  # retry-while inpfile is None and count < 10:
            count += 1
            try:
                with open(r"{}".format(self.flagpath), "rb") as inpfile:
                    try:
                        time.sleep(self.delay)
                        flags = pickle.load(inpfile)
                    except EOFError:
                        self.debug_log("{} Flags Busy: Reusing old".format(type(self).__name__))
                        flags = self.flags
                    self.flags = flags
                    self.debug_log("{} Flags Read: {}".format(type(self).__name__, self.flags))
                    return self.flags
            except FileNotFoundError:
                if count > 10:
                    break
                else:
                    time.sleep(self.delay)

    def io_flags(self):
        self.send_flags()
        self.flags = self.read_flags()

    def init_ramdisk(self):
        flagfile = ".flags.pkl"
        if sys.platform == "darwin":
            ramdisk = "/Volumes/RAMDisk"
            if not self.subprogram:
                os.system("./libs/scripts/RAMDisk mount")
                time.sleep(self.delay)  # Give the OS time to finish
        else:
            ramdisk = "/dev/shm"
        flagpath = os.path.join(ramdisk, flagfile)
        return flagpath

    def cleanup_ramdisk(self):
        if sys.platform == "darwin":
            os.system("./libs/scripts/RAMDisk unmount")
        else:
            pass


class Flags(dict):
    """Allows you to set and get {key, value} pairs like attributes"""
    def __init__(self, defaults=True):
        self.defaults = defaults
        if self.defaults:
            self.get_defaults()

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        pass

    def get_defaults(self):
        self.train = False
        self.savepb = False
        self.demo = ''
        self.fbf = ''
        self.trainer = 'rmsprop'
        self.momentum = 0.0
        self.keep = 20
        self.batch = 16
        self.epoch = 1
        self.save = 16000
        self.lr = 1e-5
        self.clip = False
        self.saveVideo = True
        self.queue = 1
        self.lb = 0.0
        self.pbLoad = ''
        self.metaLoad = ''
        self.load = -1
        self.model = ''
        self.capdevs = []
        self.json = False
        self.gpu = 0.0
        self.gpuName = '/gpu:0'
        self.threshold = 0.4
        self.verbalise = True
        self.kill = False
        self.killed = False
        self.started = False
        self.done = False
        self.error = ""
        self.progress = 0.0
        self.size = 0
        self.imgdir = './data/sample_img/'  # These paths are relative to slgrSuite.py
        self.binary = './data/bin/'
        self.config = './data/cfg/'
        self.dataset = './data/committedframes/'
        self.backup = './data/ckpt/'
        self.labels = './data/predefined_classes.txt'
        self.log = './data/logs/flow.log'
        self.annotation = './data/committedframes/'
        self.summary = './data/summaries/'


