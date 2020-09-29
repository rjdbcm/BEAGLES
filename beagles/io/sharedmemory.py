from subprocess import Popen, PIPE, STDOUT
from contextlib import contextmanager
import sys
import os

MAC_PATH = "/Volumes/RAMDisk"
LINUX_PATH = "/dev/shm"


class SharedMemory:
    """Stateful interface for shared memory on mac or linux"""
    def __init__(self):
        if sys.platform == 'darwin':
            self._path = MAC_PATH
            self._mount_point = str()
            self._info = dict()
            self.mount = self._mount
            self.unmount = self._unmount
        else:
            self._path = LINUX_PATH
            self._mount_point = self._path
            self._info = {self.__class__.__name__: self._path}
            self.mount = self._noop
            self.unmount = self._noop

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
        return self._path

    @property
    def info(self):
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
        return os.path.ismount(self._path)

    def _noop(self):
        pass

    def _mount(self):
        self.info.clear()
        proc = Popen(['./beagles/scripts/RAMDisk', 'mount'], stdout=PIPE, stderr=STDOUT, text=True)
        stdout, stderr = proc.communicate()
        if proc.poll():
            print(stdout)

    def _unmount(self):
        proc = Popen(['./beagles/scripts/RAMDisk', 'unmount'], stdout=PIPE, stderr=STDOUT, text=True)
        stdout, stderr = proc.communicate()
        if proc.poll():
            print(stdout)
        self.info.clear()

