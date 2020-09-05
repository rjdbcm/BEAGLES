from __future__ import annotations
import tensorflow as tf
import os
import sys
from libs.backend.dark import darknet
from libs.backend.base import SubsystemFactory, Subsystem, register_subsystem
import numpy as np
from os.path import basename
from libs.constants import WEIGHTS_FILE_KEYS, WGT_EXT


class Loader(SubsystemFactory):
    """
    SubsystemFactory to work with both .weights and .ckpt files
    in loading / recollecting / resolving mode
    """
    src_key = list()
    vals = list()
    VAR_LAYER = ['convolutional', 'connected', 'local', 'select', 'conv-select',
                 'extract', 'conv-extract']

    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None:
                return val
        return None
    
    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

    @classmethod
    def create(cls, path, cfg=None) -> Loader(Subsystem):
        types = dict()
        for subclass in cls.__subclasses__():
            types.update(subclass.token)
        token = os.path.splitext(path)[1] if path else WGT_EXT
        this = types.get(token, None)
        return this(cls.create_key, path, cfg)


@register_subsystem('.weights', Loader)
class WeightsLoader(Subsystem):
    """one who understands .weights files"""

    def constructor(self, path, src_layers):
        self.src_layers = src_layers
        walker = WeightsWalker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER:
                continue
            self.src_key.append([layer])
            
            if walker.eof:
                new = None
            else: 
                args = layer.signature
                new = darknet.create_darkop(*args)
            self.vals.append(new)

            if new is None:
                continue
            order = WEIGHTS_FILE_KEYS[new.type]
            for par in order:
                if par not in new.wshape:
                    continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)

        if walker.path is not None:
            assert walker.offset == walker.size, \
            'expect {} bytes, found {}'.format(
                walker.offset, walker.size)
            print('Successfully identified {} bytes'.format(
                walker.offset))


@register_subsystem('', Loader)
class CheckpointLoader(Subsystem):
    """
    one who understands .ckpt files, very much
    """
    def constructor(self, ckpt, ignore):
        meta = ckpt + '.meta'
        with tf.Graph().as_default():
            with tf.compat.v1.Session().as_default() as sess:
                saver = tf.compat.v1.train.import_meta_graph(meta)
                saver.restore(sess, ckpt)
                for var in tf.compat.v1.global_variables():
                    name = var.name.split(':')[0]
                    packet = [name, var.get_shape().as_list()]
                    self.src_key += [packet]
                    self.vals += [var.eval(sess)]


class WeightsWalker(object):
    """incremental reader of float32 binary files"""
    def __init__(self, path):
        self.eof = False  # end of file
        self.path = path  # current pos
        if path is None: 
            self.eof = True
            return
        else: 
            self.size = os.path.getsize(path)# save the path
            major, minor, revision, seen = np.memmap(path, shape=(), mode='r', offset=0, dtype='({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def walk(self, size):
        if self.eof:
            return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, \
            'Over-read {}'.format(self.path)
        float32_1D_array = np.memmap(self.path, shape=(), mode='r', offset=self.offset,
                                     dtype='({})float32,'.format(size))

        self.offset = end_point
        if end_point == self.size: 
            self.eof = True
        return float32_1D_array



