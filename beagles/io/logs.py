import os
import logging
import inspect
from logging.handlers import RotatingFileHandler

LOGDIR = './data/logs/flow.log'

FORMAT = logging.Formatter('{asctime} | {levelname:7} | {name:<13} | {funcName:<20} | {message}', style='{')

logging.captureWarnings(True)


def get_logger(level=logging.INFO):
    try:
        caller = inspect.stack()[1][0].f_locals["self"].__class__.__name__
    except KeyError:
        caller = 'None'
    logger = logging.getLogger(caller)
    logger.findCaller()
    logger.handlers = []
    root, file = os.path.splitext(LOGDIR)
    handler = ''.join([root, file])
    logfile = RotatingFileHandler(handler, backupCount=20)
    logfile.setFormatter(FORMAT)
    logger.addHandler(logfile)
    logger.setLevel(level)
    logger.propagate = False
    return logger, logfile