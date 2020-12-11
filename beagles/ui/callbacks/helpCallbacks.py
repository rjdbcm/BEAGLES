import sys
import webbrowser
from cython import __version__ as cy_version
from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt5.QtWidgets import QMessageBox
from tensorflow import version as tfVersion
from av import __version__ as pyavVersion
from numpy import __version__ as npVersion
from matplotlib import __version__ as pltVersion
import traces
from pandas import __version__ as pdVersion
from qdarkstyle import __version__ as qdarkVersion
from PIL import __version__ as pilVersion
from scipy import __version__ as scipyVersion
from defusedxml import __version__ as defusedVersion
from beagles.base.constants import APP_NAME
from beagles.ui.functions.helpFunctions import HelpFunctions
from beagles.base.version import __version__

pyVersion = '.'.join([str(i) for i in sys.version_info[:3]])
msg = f"""
{APP_NAME} v{__version__} Version Info:
Cython v{cy_version}
defusedxml v{defusedVersion}
matplotlib v{pltVersion}
Numpy v{npVersion}
Pandas v{pdVersion}
PIL v{pilVersion}
PyAV v{pyavVersion}
PyQt v{PYQT_VERSION_STR}
Python v{pyVersion}
QDarkStyle v{qdarkVersion}
Qt v{QT_VERSION_STR}
SciPy v{scipyVersion}
Tensorflow v{tfVersion.VERSION}
Traces v{traces.__version__}
"""


class HelpCallbacks(HelpFunctions):

    def showInfo(self):
        # noinspection PyTypeChecker
        QMessageBox.information(self, u'Information', msg)

    def showTutorialDialog(self):
        webbrowser.open_new_tab("https://youtu.be/p0nR2YsCY_U")