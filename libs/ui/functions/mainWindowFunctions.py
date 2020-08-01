import os
import sys
import platform
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QWidget, QMessageBox, QListWidgetItem
from libs.constants import *
from libs.qtUtils import newIcon
from libs.labelFile import LabelFile
from libs.labelDialog import LabelDialog
from libs.utils.flags import Flags


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