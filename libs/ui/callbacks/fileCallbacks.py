import os
import shutil
from libs.labelFile import LabelFile
from libs.constants import *
from libs.ui.functions.fileFunctions import FileFunctions
# noinspection PyUnresolvedReferences
from PyQt5.QtCore import QProcess
from PyQt5.QtGui import QImageReader
from PyQt5.QtWidgets import QFileDialog


class FileCallbacks(FileFunctions):

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def changeFormat(self):
        if self.usingPascalVocFormat:
            self.setFormat(FORMAT_YOLO)
        elif self.usingYoloFormat:
            self.setFormat(FORMAT_PASCALVOC)

    def impVideo(self, _value=False):
        if not self.mayContinue():
            return
        path = self.setDefaultOpenDirPath()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        formats = ['*.avi', '*.mp4', '*.wmv', '*.mkv', '*.mpeg']
        filters = "Video Files (%s)" % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % APP_NAME,
            path, filters, options=options)
        target = os.path.join(
            self.rawframesDataPath,
            os.path.basename(os.path.splitext(filename[0])[0]))
        if not os.path.exists(target):
            os.makedirs(target)
        if filename[0] != '':
            if isinstance(filename, (tuple, list)):
                video = shutil.copy2(filename[0], target)
                self.logger.info('Extracting frames from {} to {}'.format(
                    filename, target))
                self.frameCapture(video)
                self.importDirImages(target)
        if target is not None and len(target) > 1:
            self.defaultSaveDir = target
        else:
            pass

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(str(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower()
                   for fmt in QImageReader.supportedImageFormats()]
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filters = "Image files (%s)" % ' '.join(formats)
        # noinspection PyTypeChecker
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image file' % APP_NAME, path, filters,
            options=options)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def openDir(self, _value=False):
        if not self.mayContinue():
            return
        path = self.setDefaultOpenDirPath()
        targetDirPath = str(QFileDialog.getExistingDirectory(
            self, '%s - Open Directory' % APP_NAME, path,
                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        # set the annotation save directory to the target directory
        self.defaultSaveDir = targetDirPath if targetDirPath != "" else \
            self.defaultSaveDir
        self.importDirImages(targetDirPath)

    def changeSaveDir(self, _value=False):
        if self.defaultSaveDir is not None:
            path = str(self.defaultSaveDir)
        else:
            path = '../..'

        dirpath = str(QFileDialog.getExistingDirectory(
            self, '%s - Save annotations to the directory' % APP_NAME, path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage(
            '%s . Annotation will be saved to %s' %
            ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotation(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(str(self.filePath)) \
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename = str(QFileDialog.getOpenFileName(
                self, '%s - Choose a xml file' % APP_NAME, path, filters,
                options=options))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(str(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(str(self.defaultSaveDir),
                                         savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())
