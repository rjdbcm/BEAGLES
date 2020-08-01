import sys
import subprocess
import cv2
from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt5.QtWidgets import QMessageBox
from tensorflow import version as tensorflow_version
from libs.constants import *
from libs.ui.functions.helpFunctions import HelpFunctions
from libs.version import __version__


class HelpCallbacks(HelpFunctions):
    def showInfo(self):
        msg_string = u'{0}\n' \
                     u'App Version: {1}\n' \
                     u'Python Version: {2}.{3}.{4}\n' \
                     u'Qt Version: {5}\n' \
                     u'PyQt Version: {6}\n' \
                     u'Tensorflow Version: {7}\n'\
                     u'Tensorflow Compiler Version:\n{8}\n'\
                     u'OpenCV Version: {9}'
        msg = msg_string.format(APP_NAME,
                                __version__,
                                *sys.version_info[:3],
                                QT_VERSION_STR,
                                PYQT_VERSION_STR,
                                tensorflow_version.VERSION,
                                tensorflow_version.COMPILER_VERSION,
                                cv2.__version__)
        # noinspection PyTypeChecker
        QMessageBox.information(self, u'Information', msg)

    def showTutorialDialog(self):
        link = ["https://youtu.be/p0nR2YsCY_U"]
        subprocess.Popen(self.getAvailableScreencastViewer() + link)
