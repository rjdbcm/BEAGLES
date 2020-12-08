import sys
import webbrowser
from cython import __version__ as cy_version
from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt5.QtWidgets import QMessageBox
from tensorflow import version as tfVersion
import traces
from beagles.base.constants import APP_NAME
from beagles.ui.functions.helpFunctions import HelpFunctions
from beagles.base.version import __version__

pyVersion = '.'.join([str(i) for i in sys.version_info[:3]])
msg = f"""
{APP_NAME} Version: {__version__}
Python Version: {pyVersion}
Qt Version: {QT_VERSION_STR}
PyQt Version: {PYQT_VERSION_STR}
Tensorflow Version: {tfVersion.VERSION}
Tensorflow Compiler Version: 
{tfVersion.COMPILER_VERSION}
Cython Version: {cy_version}
Traces Version: {traces.__version__}
"""


class HelpCallbacks(HelpFunctions):

    def showInfo(self):
        # noinspection PyTypeChecker
        QMessageBox.information(self, u'Information', msg)

    def showTutorialDialog(self):
        webbrowser.open_new_tab("https://youtu.be/p0nR2YsCY_U")