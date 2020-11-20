import os
import sys
import logging
import inspect
from subprocess import DEVNULL
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
from beagles.io.flags import Flags

FORMAT = logging.Formatter(
    '{asctime} | {levelname:7} | {name:<13} | {funcName:<20} |'
    ' {message}', style='{')

logging.captureWarnings(True)


def get_logger(level=logging.INFO):
    try:
        caller = inspect.stack()[1][0].f_locals["self"].__class__.__name__
    except KeyError:
        caller = 'None'
    logger = logging.getLogger(caller)
    logger.handlers = []
    root, file = os.path.splitext(Flags().log)
    handler = ''.join([root, file])
    logfile = RotatingFileHandler(handler, backupCount=20)
    logfile.setFormatter(FORMAT)
    tf_logfile = RotatingFileHandler('.tf'.join([root, file]), backupCount=20)
    tf_logfile.setFormatter(FORMAT)
    # don't re-add the same handler
    logger.addHandler(logfile)
    logger.addHandler(tf_logfile)
    logger.setLevel(level)
    logger.propagate = False
    return logger