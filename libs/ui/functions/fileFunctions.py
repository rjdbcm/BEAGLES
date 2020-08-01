import os
import cv2
import sys
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtWidgets import QListWidgetItem
from libs.constants import *
from libs.labelFile import LabelFile, LabelFileError
from libs.pascalVoc import XML_EXT, PascalVocReader
from libs.ui.functions.mainWindowFunctions import MainWindowFunctions
from libs.utils.flags import Flags
from libs.yolo import TXT_EXT, YoloReader


class FileFunctions(MainWindowFunctions):
    def __init__(self): super(FileFunctions, self).__init__()

    @staticmethod
    def frameCapture(path):
        vidObj = cv2.VideoCapture(path)
        count = 1
        success = 1
        name = os.path.splitext(path)[0]
        total_zeros = len(str(int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))))
        while success:
            success, image = vidObj.read()
            fileno = str(count)
            cv2.imwrite(
                "{}_frame_{}.jpg".format(name, fileno.zfill(total_zeros)),
                image)
            count += 1

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = str(filePath)
        unicodeFilePath = str(filePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            try:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
                self.fileListWidget.scrollToItem(fileWidgetItem)
            except ValueError:
                pass

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(
                        u'Error opening file',
                        (u"<p><b>%s</b></p>"
                         u"<p>Make sure <i>%s</i> is a valid label file.")
                        % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(
                    u'Error opening file',
                    u"<p>Make sure <i>%s</i> is a valid image file."
                    % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(self.filePath)[0])
                xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

                """Annotation file priority:
                PascalXML > YOLO
                """
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)
            else:
                xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)

            self.setWindowTitle(APP_NAME + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(
                    self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(
                    self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.setFormat(FORMAT_PASCALVOC)

        pascal_voc_reader = PascalVocReader(xmlPath)
        shapes = pascal_voc_reader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = pascal_voc_reader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.setFormat(FORMAT_YOLO)
        yolo_reader = YoloReader(txtPath, self.image)
        shapes = yolo_reader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = yolo_reader.verified


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return default
