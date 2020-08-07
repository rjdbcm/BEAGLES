from PyQt5.QtCore import QJsonDocument, QFile
import pickle
import os
from libs.constants import SAVE_SETTINGS_PATH
# ToDo: Deprecate and add JSON interface


class Settings(object):
    __data = {}

    def __init__(self):
        self.__dict__ = self.__data

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        return default

    def save(self):
        if SAVE_SETTINGS_PATH:
            print(self.__dict__)
            with open(SAVE_SETTINGS_PATH, 'wb') as f:
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
                return True
        return False

    def load(self):
        try:
            if os.path.exists(SAVE_SETTINGS_PATH):
                with open(SAVE_SETTINGS_PATH, 'rb') as f:
                    self.__dict__ = pickle.load(f)
                    return True
        except Exception as e:
            print('Loading setting failed: ', e)
        return False

    def reset(self):
        if os.path.exists(SAVE_SETTINGS_PATH):
            os.remove(SAVE_SETTINGS_PATH)
            print('Remove setting pkl file ${0}'.format(SAVE_SETTINGS_PATH))
        self.__dict__ = self.__data
