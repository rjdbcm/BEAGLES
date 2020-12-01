import os
import sys
sys.path.append(os.getcwd())
from beagles.io.flags import SharedFlagIO
from beagles.backend.net import NetBuilder

if __name__ == '__main__':
    io = SharedFlagIO(subprogram=False)
    flags = io.read_flags()
    flags.started = True
    net_builder = NetBuilder(flags=flags)
    flags = io.read_flags()
    if flags.train:
        net_builder.train()
    elif flags.video:
        net_builder.annotate()
    else:
        net_builder.predict()
    flags = io.read_flags()
    flags.progress = 100.0
    flags.done = True
    io.io_flags()