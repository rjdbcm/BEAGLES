import os
import sys
import time
import inspect
from json.decoder import JSONDecodeError
import logging
import subprocess
import logging.handlers
from libs.utils.flags import Flags
#  TODO: extract logging and ramdisk


class FlagIO(object):
    """Base object for logging and shared memory flag read/write operations"""

    def __init__(self, subprogram=False):
        self.subprogram = subprogram
        self.flags = Flags()

        logging.captureWarnings(True)
        try:
            caller = inspect.stack()[1][0].f_locals["self"].__class__.__name__
        except KeyError:
            caller = 'BackendWrapper'
        self.logger = logging.getLogger(caller)
        formatter = logging.Formatter(
            '{asctime} | {levelname:7} | {name:<11} | {funcName:<20} |'
            ' {message}', style='{')
        self.logfile = logging.handlers.RotatingFileHandler(self.flags.log,
                                                            backupCount=20)
        self.tf_logfile = logging.handlers.RotatingFileHandler(
            os.path.splitext(Flags().log)[0] + ".tf" +
            os.path.splitext(Flags().log)[1], backupCount=20)

        self.logfile.setFormatter(formatter)
        self.tf_logfile.setFormatter(formatter)
        # don't re-add the same handler
        if not str(self.logfile) in str(self.logger.handlers):
            self.logger.addHandler(self.logfile)

        self.flagpath = self.init_ramdisk()

        try:
            if self.read_flags().cli:
                self.logstream = logging.StreamHandler()
                self.logstream.setFormatter(formatter)
                if not str(self.logstream) in str(self.logger.handlers):
                    self.logger.addHandler(self.logstream)
        except AttributeError:
            pass

        if subprogram:
            self.read_flags()
            try:
                if self.flags.verbalise:
                    self.logger.setLevel(logging.DEBUG)
                else:
                    self.logger.setLevel(logging.INFO)
            except AttributeError:
                self.logger.setLevel(logging.DEBUG)
            try:
                f = open(self.flagpath)
                f.close()
            except FileNotFoundError:
                time.sleep(1)

    def send_flags(self):
        self.logger.debug(self.flags)
        try:
            with open(r"{}".format(self.flagpath), "w") as outfile:
                self.flags.to_json(outfile)
        except FileNotFoundError:
            pass

    def read_flags(self):
        file = None
        count = 0
        while file is None:  # retry-while file is None and count < 10:
            count += 1
            try:
                with open(r"{}".format(self.flagpath), "r") as file:
                    try:
                        flags = self.flags.from_json(file)
                    except JSONDecodeError as e:
                        if not e.pos:  # char 0 == flags busy
                            self.logger.warning("Flags Busy: Reusing old")
                            flags = self.flags
                        else:
                            raise
                    self.flags = flags
                    self.logger.debug(self.flags)
                    return self.flags
            except FileNotFoundError:
                if count > 10:
                    break

    def io_flags(self):
        self.send_flags()
        self.flags = self.read_flags()

    def init_ramdisk(self):
        flagfile = ".flags.json"
        if sys.platform == "darwin":
            ramdisk = "/Volumes/RAMDisk"
            if not self.subprogram:
                proc = subprocess.Popen(['./libs/scripts/RAMDisk', 'mount'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                stdout, stderr = proc.communicate()
                for line in stdout.decode('utf-8').splitlines():
                    self.logger.info(line)
                if stderr:
                    self.logger.debug(stderr.decode('utf-8'))
        else:
            ramdisk = "/dev/shm"
        flagpath = os.path.join(ramdisk, flagfile)
        return flagpath

    def cleanup_ramdisk(self):
        if sys.platform == "darwin":
            proc = subprocess.Popen(['./libs/scripts/RAMDisk', 'unmount'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            stdout, stderr = proc.communicate()
            for line in stdout.decode('utf-8').splitlines():
                self.logger.info(line)
            if stderr:
                self.logger.debug(stderr.decode('utf-8'))
        else:
            os.remove(self.flagpath)
