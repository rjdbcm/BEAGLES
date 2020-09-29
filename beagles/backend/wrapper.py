#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd())
from beagles.backend.trainer import Trainer
from beagles.backend.predictor import Predictor
from beagles.backend.annotator import Annotator
from beagles.io.flags import  SharedFlagIO
from beagles.base.flags import Flags

if __name__ == '__main__':
    io = SharedFlagIO(subprogram=True)
    flags = io.read_flags()
    flags.started = True
    if flags.train:
        wrapper = Trainer(flags)
    elif flags.video != []:
        wrapper = Annotator(flags)
    else:
        wrapper = Predictor(flags)
    wrapper()
    io.read_flags()
    flags.progress = 100
    flags.done = True
    io.io_flags()
    exit(0)
