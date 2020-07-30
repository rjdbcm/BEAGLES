import os
import sys
import subprocess
from tensorflow import version as tf_version
# noinspection PyUnresolvedReferences
from PyQt5.QtGui import *
# noinspection PyUnresolvedReferences
from PyQt5.QtCore import *
# noinspection PyUnresolvedReferences
from PyQt5.QtWidgets import *
from libs.constants import *
from libs.ui.callbacks.fileAndFolderCallbacks import FileAndFolderCallbacks
from libs.ui.callbacks.boundingBoxCallbacks import BoundingBoxCallbacks
from libs.labelFile import LabelFile
from libs.version import __version__


class MainWindowCallbacks(FileAndFolderCallbacks, BoundingBoxCallbacks):

    def nextImg(self, _value=False):
        # Proceeding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSaveDir()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def prevImg(self, _value=False):
        # Proceeding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSaveDir()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def verifyImg(self, _value=False):
        # Proceeding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.labelFile = LabelFile()
                self.logger.info(self.labelFile)
                if self.labelFile is not None:
                    self.labelFile.toggleVerify()
                else:
                    return
            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def changeFormat(self):
        if self.usingPascalVocFormat:
            self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat:
            self.set_format(FORMAT_PASCALVOC)

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def visualize(self):
        subprocess.Popen(self.screencastViewer +
                         ['http://localhost:6006/#scalars&_smoothingWeight=0'])

    def trainModel(self):
        if not self.mayContinue():
            return
        self.trainDialog.show()

    def advancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.setCreateMode.setEnabled(True)
            self.actions.setEditMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def showInfo(self):
        msg = u'{0}\n' \
              u'App Version: {1}\n' \
              u'Python Version: {2}.{3}.{4}\n' \
              u'Qt Version: {5}\n' \
              u'PyQt Version: {6}\n' \
              u'Tensorflow Version: {7}'.format(
            APP_NAME,
            __version__,
            *sys.version_info[:3],
            QT_VERSION_STR,
            PYQT_VERSION_STR,
            tf_version.VERSION)
        QMessageBox.information(self, u'Information', msg)

    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def zoomIn(self):
        self.addZoom(10)

    def zoomOut(self):
        self.addZoom(-10)

    def zoomOrg(self):
        self.setZoom(100)

    def setFitWin(self, value=True):
        if value:
            self.actions.setFitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.setFitWin.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()


