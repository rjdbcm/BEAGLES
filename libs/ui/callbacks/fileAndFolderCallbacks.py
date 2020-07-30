import os
import errno
import shutil
import cv2
from libs.labelFile import LabelFile
from libs.constants import *
# noinspection PyUnresolvedReferences
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImageReader



class FileAndFolderCallbacks:
    def impVideo(self, _value=False):
        if not self.mayContinue():
            return
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) \
                if self.filePath else '.'
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        formats = ['*.avi', '*.mp4', '*.wmv', '*.mkv', '*.mpeg']
        filters = "Video Files (%s)" % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % APP_NAME,
            defaultOpenDirPath, filters, options=options)
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
                frame_capture(video)
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
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image file' % APP_NAME, path, filters,
            options=options)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def openDir(self, _value=False):
        if not self.mayContinue():
            return

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if \
                self.filePath else '.'
        targetDirPath = str(QFileDialog.getExistingDirectory(
            self, '%s - Open Directory' % APP_NAME, defaultOpenDirPath,
                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        # set the annotation save directory to the target directory
        self.defaultSaveDir = targetDirPath if targetDirPath != "" else \
            self.defaultSaveDir
        print(self.defaultSaveDir)
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

    def commitAnnotatedFrames(self):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure you want to commit all "
                                     "open files?", QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.No or not self.mayContinue():
            return
        else:
            pass

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if \
                self.filePath else '.'
        if defaultOpenDirPath == self.committedframesDataPath:
            self.errorMessage("", "These files are already committed.")
            return

        filelist = []
        for file in os.listdir(defaultOpenDirPath):
            filename = os.fsdecode(file)
            if filename.endswith(".xml"):
                self.logger.info(
                    "Moving {0} to data/committedframes/{0}".format(filename))
                filename = os.path.join(defaultOpenDirPath, filename)
                basename = os.path.splitext(filename)[0]
                filelist.append(filename)
                filelist.append(basename + '.jpg')
            else:
                continue

        for i in filelist:
            dest = os.path.join(self.committedframesDataPath,
                                os.path.split(i)[1])
            try:
                os.rename(i, dest)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    shutil.copy2(i, dest)
                    os.remove(i)
                else:
                    raise

        self.importDirImages(defaultOpenDirPath)


def frame_capture(path):
    vidObj = cv2.VideoCapture(path)
    count = 1  # Start the frame index at 1 >.>
    success = 1
    name = os.path.splitext(path)[0]
    total_zeros = len(str(int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))))
    while success:
        success, image = vidObj.read()
        fileno = str(count)
        cv2.imwrite("{}_frame_{}.jpg".format(name, fileno.zfill(total_zeros)),
                    image)
        count += 1
