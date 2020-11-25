import os
import sys
sys.path.append(os.getcwd())
from beagles.io.flags import SharedFlagIO
from beagles.backend.net import NetBuilder, predict, annotate

if __name__ == '__main__':
    io = SharedFlagIO(subprogram=True)
    flags = io.read_flags()
    flags.started = True
    net_builder = NetBuilder(flags=flags)
    net_builder()
    flags = io.read_flags()
    if flags.train:
        net_builder.train()
    elif flags.video:
        annotate(flags, net, framework)
    else:
        predict(flags, net, framework)
    flags = io.read_flags()
    flags.progress = 100.0
    flags.done = True
    io.io_flags()
    exit(0)