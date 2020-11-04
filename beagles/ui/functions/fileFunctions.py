import os
import codecs
import cv2
from functools import partial
from PyQt5.QtCore import QFileInfo, QPointF
from PyQt5.QtGui import QImage, QColor, QPixmap, QImageReader
from PyQt5.QtWidgets import QAction, QFileDialog
from beagles.base.constants import *
from beagles.io.labelFile import LabelFile, LabelFileError
from beagles.io.pascalVoc import XML_EXT, PascalVocReader
from beagles.ui.functions.mainWindowFunctions import MainWindowFunctions
from beagles.ui import newIcon
from beagles.base.shape import Shape
from beagles.io.yolo import TXT_EXT, YoloReader


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
            cv2.imwrite(f"{name}_frame_{fileno.zfill(total_zeros)}.jpg", image)
            count += 1

    @staticmethod
    def scanAllImages(folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for
                      fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = str(os.path.abspath(relativePath))
                    images.append(path)
        naturalSort(images, key=lambda x: x.lower())
        return images

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

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

                # Annotation file priority: PascalXML > YOLO
                self.loadBasedOnPriority(xmlPath, txtPath)
            else:
                xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                self.loadBasedOnPriority(xmlPath, txtPath)

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

    def loadBasedOnPriority(self, first, second):
        if os.path.isfile(first):
            self.loadPascalXMLByFilename(first)
        elif os.path.isfile(second):
            self.loadYOLOTXTByFilename(second)

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

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def saveFileDialog(self, removeExt=True):
        caption = f'{APP_NAME} - Choose File'
        filters = f'File (*{LabelFile.suffix})'
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = str(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0]
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image.
                # If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = self.generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = self.generateColorByText(label)

            self.addLabel(shape)

        self.canvas.loadShapes(s)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = str(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add different annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes,
                                                   self.filePath,
                                                   self.imageData,
                                                   self.lineColor.getRgb(),
                                                   self.fillColor.getRgb())
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes,
                                              self.filePath, self.imageData,
                                              self.labelHist,
                                              self.lineColor.getRgb(),
                                              self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath,
                                    self.imageData, self.lineColor.getRgb(),
                                    self.fillColor.getRgb())
            self.io.logger.info('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def loadPredefinedClasses(self):
        predefClassesFile = self.predefinedClasses
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return default


def naturalSort(items: list, key=lambda s: s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        def convert(text):
            return int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    items.sort(key=sort_key)