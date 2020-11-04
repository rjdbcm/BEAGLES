import os
import time
from json.decoder import JSONDecodeError
import logging
from beagles.base.flags import Flags
from beagles.io.sharedmemory import SharedMemory
from beagles.io.logs import get_logger

FLAG_FILE = ".flags.json"


class SharedFlagIO(object):
    """Base object for logging and shared memory flag read/write operations"""

    def __init__(self, flags=None, subprogram=False):
        self.subprogram = subprogram
        self.flags = flags if flags else Flags()
        self.logger = get_logger()
        self.shm = SharedMemory()
        if not self.subprogram and not self.shm.mounted:
            self.shm.mount()
        self.flag_path = os.path.join(self.shm.path, FLAG_FILE)

        if subprogram:
            self.read_flags()
            try:
                if self.flags.verbalise:
                    self.logger.setLevel(logging.DEBUG)
            except AttributeError:
                self.logger.setLevel(logging.DEBUG)
            try:
                f = open(self.flag_path)
                f.close()
            except FileNotFoundError:
                time.sleep(1)

    def send_flags(self):
        self.logger.debug(self.flags)
        try:
            with open(r"{}".format(self.flag_path), "w") as outfile:
                self.flags.to_json(outfile)
        except FileNotFoundError:
            pass

    def read_flags(self):
        file = None
        count = 0
        while file is None:  # retry-while file is None and count < 10:
            count += 1
            try:
                with open(r"{}".format(self.flag_path), "r") as file:
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

    def cleanup_flags(self):
        os.remove(self.flag_path)
        self.shm.unmount()


