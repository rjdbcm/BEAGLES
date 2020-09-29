import os
import sys
import hashlib
import platform
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtWidgets import QWidget, QMessageBox, QListWidgetItem
from beagles.base.constants import *
from beagles.ui.qtUtils import newIcon, addActions
from beagles.io.labelFile import LabelFile
from beagles.ui.widgets.labelDialog import LabelDialog
from beagles.base.flags import Flags


class MainWindowFunctions(QWidget, QObject):
    def __init__(self):
        super(MainWindowFunctions, self).__init__()
        self.labelHist = []
        # Application state.
        self.image = QImage()
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        self.difficult = False
        self.lastOpenDir = None
        self.defaultSaveDir = str()
        if hasattr(sys, 'frozen'):
            self.rawframesDataPath = os.path.join(
                os.path.dirname(sys.executable), 'data/rawframes/')
            self.committedframesDataPath = os.path.join(
                os.path.dirname(sys.executable), Flags().dataset)
        else:
            self.rawframesDataPath = os.path.abspath('./data/rawframes/')
            self.committedframesDataPath = os.path.abspath(Flags().dataset)
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

    @staticmethod
    def getAvailableScreencastViewer():
        os_name = platform.system()
        if os_name == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif os_name == 'Linux':
            return ['xdg-open']
        elif os_name == 'Darwin':
            return ['open', '-a', 'Safari']

    # noinspection PyMethodMayBeStatic
    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def setDirty(self):
        self.dirty = True
        # noinspection PyUnresolvedReferences
        self.actions.saveFile.setEnabled(True)

    def setClean(self):
        self.dirty = False
        # noinspection PyUnresolvedReferences
        self.actions.saveFile.setEnabled(False)
        # noinspection PyUnresolvedReferences
        self.actions.create.setEnabled(True)

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        # noinspection PyTypeChecker
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        # noinspection PyTypeChecker
        return QMessageBox.critical(self, title, '<p><b>%s</b></p>%s' % (title, message))

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def setFormat(self, annotation_format):
        if annotation_format == FORMAT_PASCALVOC:
            # noinspection PyUnresolvedReferences
            self.actions.changeFormat.setText(FORMAT_PASCALVOC)
            # noinspection PyUnresolvedReferences
            self.actions.changeFormat.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif annotation_format == FORMAT_YOLO:
            # noinspection PyUnresolvedReferences
            self.actions.changeFormat.setText(FORMAT_YOLO)
            # noinspection PyUnresolvedReferences
            self.actions.changeFormat.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def setDefaultOpenDirPath(self):
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) \
                if self.filePath else '.'
        return defaultOpenDirPath

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.nextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(os.path.basename(imgPath))
            self.fileListWidget.addItem(item)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        item = os.path.join(self.dirname, item.text())
        currIndex = self.mImgList.index(item)
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)
    @staticmethod
    def generateColorByText(text):
        s = str(text)
        hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
        r = int((hashCode / 255) % 255)
        g = int((hashCode / 65025) % 255)
        b = int((hashCode / 16581375) % 255)
        return QColor(r, g, b, 100)
