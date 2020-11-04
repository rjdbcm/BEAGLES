import os
import json
import codecs
from PyQt5.QtCore import QSize, QPoint, QByteArray
from PyQt5.QtGui import QColor
from beagles.base.constants import SAVE_SETTINGS_PATH


class Settings(object):
    __data = {}

    trans_reverse = {
        "<class 'PyQt5.QtCore.QSize'>": QSize,
        "<class 'PyQt5.QtCore.QPoint'>": QPoint,
        "<class 'PyQt5.QtCore.QByteArray'>": QByteArray,
        "<class 'PyQt5.QtGui.QColor'>": QColor
    }

    def __init__(self):
        self.__dict__ = self.__data

    def __setitem__(self, key, value):
        """store PyQt5 Objects as tuple(str(obj.__class__), *args)"""
        if isinstance(value, QSize):
            value = (str(value.__class__), value.width(), value.height())
        elif isinstance(value, QPoint):
            value = (str(value.__class__), value.x(), value.y())
        elif isinstance(value, QByteArray):
            value = (str(value.__class__), str(value.data()).strip("b'"))
        elif isinstance(value, QColor):
            value = (str(value.__class__),) + value.getRgb()
        self.__dict__[key] = value

    def __getitem__(self, key):
        value = self.__dict__[key]
        if isinstance(value, list):
            try:
                class_ = self.trans_reverse[value[0]]
                if key.endswith('/state'):
                    value = class_(bytearray(codecs.encode(value[1])))
                elif key.endswith('/color'):
                    value = class_.fromRgb(*[int(i) for i in value[1:]])
                elif key.endswith(('/position', '/size')):
                    value = class_(*[int(i) for i in value[1:]])
                else:
                    value = class_(*value)
            except KeyError:
                pass
        return value

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__getitem__(key)
        return default

    def save(self):
        if SAVE_SETTINGS_PATH:
            with open(SAVE_SETTINGS_PATH, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
                return True
        return False

    def load(self):
        try:
            if os.path.exists(SAVE_SETTINGS_PATH):
                with open(SAVE_SETTINGS_PATH, 'r') as f:
                    self.__dict__.update(dict(json.load(f)))
                    return True
        except Exception as e:
            print(f"Failed to load setting {e}")

        return False

    def reset(self):
        if os.path.exists(SAVE_SETTINGS_PATH):
            os.remove(SAVE_SETTINGS_PATH)
        self.__dict__ = self.__data

