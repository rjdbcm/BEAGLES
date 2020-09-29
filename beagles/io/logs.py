import os
import logging
import inspect
from logging.handlers import RotatingFileHandler
from beagles.io.flags import Flags

FORMAT = logging.Formatter(
    '{asctime} | {levelname:7} | {name:<11} | {funcName:<20} |'
    ' {message}', style='{')

logging.captureWarnings(True)


def get_logger(level=logging.INFO):
    try:
        caller = inspect.stack()[1][0].f_locals["self"].__class__.__name__
    except KeyError:
        caller = 'None'
    logger = logging.getLogger(caller)

    root, file = os.path.splitext(Flags().log)

    handler = ''.join([root, file])
    logfile = RotatingFileHandler(handler, backupCount=20)
    logfile.setFormatter(FORMAT)

    tf_handler = '.tf'.join([root, file])
    tf_logfile = RotatingFileHandler(tf_handler, backupCount=20)
    tf_logfile.setFormatter(FORMAT)
    # don't re-add the same handler
    if not str(logfile) in str(logger.handlers):
        logger.addHandler(logfile)

    logger.setLevel(level)

    return logger