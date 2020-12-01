import numpy as np
from abc import abstractmethod


class Layer(object):
    """Parent class for all darknet layers."""
    def __init__(self, *args):
        self.type, self.number, *setup_args = self._signature = list(args)
        self.w = dict()  # weights
        self.h = dict()  # placeholders
        self.wshape = dict()  # weight shape
        self.wsize = dict()  # weight size
        self.setup(*setup_args)  # set attr up
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    @property
    def signature(self):
        return self._signature

    # For comparing two layers
    def __eq__(self, other):
        return self.signature == other.signature

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self._signature)

    def __str__(self):
        return str(self._signature)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w):
        self.w = w

    def present(self):
        self.presenter = self

    @abstractmethod
    def setup(self, *args):
        pass

    @abstractmethod
    def finalize(self, *args):
        pass
