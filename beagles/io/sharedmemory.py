from subprocess import Popen, PIPE, STDOUT
from contextlib import contextmanager
import sys
import os
from beagles.io.logs import get_logger

MAC_PATH = "/Volumes/RAMDisk"
LINUX_PATH = "/dev/shm"


class SharedMemory:
    """Stateful interface for shared memory on mac or linux"""
    def __init__(self):
        self.log, self.logfile = get_logger()
        if sys.platform == 'darwin':
            self._path = MAC_PATH
            self._mount_point = str()
            self._info = dict()
        else:
            self._path = LINUX_PATH
            self._mount_point = self._path
            self._info = {self.__class__.__name__: self._path}

    @contextmanager
    def __call__(self):
        try:
            yield self.mount()
        finally:
            self.unmount()

    def __repr__(self):
        data = list()
        for k, v in self.info.items():
            data += [": ".join([k, v])]
        return '\n'.join(data)

    @property
    def path(self):
        """:obj:`str`: Path to the shared memory drive."""
        return self._path

    @property
    def info(self):
        """:obj:`dict`: Info about the shared memory drive."""
        if not self._info:
            proc = Popen("diskutil info RAMDisk".split(' '), stdout=PIPE, stderr=STDOUT)
            stdout, stderr = proc.communicate()
            for line in stdout.decode('utf8').splitlines():
                i = ([i.strip(' ') for i in line.split(':', maxsplit=2)])
                if len(i) == 2:
                    if not proc.poll():
                        self._info.update({i[0]: i[1]})
        return self._info

    @property
    def mounted(self):
        """:obj:`bool`: `True` if the shared memory drive is mounted otherwise false."""
        return os.path.ismount(self._path)

    def mount(self):
        """Mounts a shared memory drive using
        :ref:`RAMDisk mount <ramdisk-ref>` if MacOS otherwise no operation.
        """
        if sys.platform == 'darwin':
            return self._mount()
        else:
            return self._noop()

    def unmount(self):
        """Unmounts a shared memory drive using
        :ref:`RAMDisk unmount <ramdisk-ref>` if MacOS otherwise no operation.
        """
        if sys.platform == 'darwin':
            return self._unmount()
        else:
            return self._noop()

    def _noop(self):
        pass

    def _mount(self):
        self.info.clear()
        proc = Popen(['./beagles/scripts/RAMDisk', 'mount'], stdout=PIPE, stderr=STDOUT, text=True)
        stdout, stderr = proc.communicate()
        [self.log.info(line) for line in stdout.splitlines()]
        [self.log.info(line) for line in self.__repr__().splitlines()]



    def _unmount(self):
        proc = Popen(['./beagles/scripts/RAMDisk', 'unmount'], stdout=PIPE, stderr=STDOUT, text=True)
        stdout, stderr = proc.communicate()
        [self.log.info(line) for line in stdout.splitlines()]
        self.info.clear()

