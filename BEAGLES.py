#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import random
import os.path
import argparse
import platform
from functools import partial

# Add internal libs
# noinspection PyUnresolvedReferences
from libs.resources import *
from libs.ui.BEAGLES import BeaglesMainWindow, getStr
from libs.constants import *
from libs.qtUtils import *
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.labelDialog import LabelDialog
from libs.utils.flags import Flags, FlagIO
from libs.backend import FlowDialog
from libs.colorDialog import ColorDialog
from libs.project import ProjectDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.pascalVoc import XML_EXT
from libs.yolo import TXT_EXT
from libs.hashableQListWidgetItem import HashableQListWidgetItem


class MainWindow(BeaglesMainWindow):

    # noinspection PyShadowingBuiltins
    def __init__(self, filename=None, predefined_class_file=None,
                 save_directory=None):
        super(MainWindow, self).__init__()
        self.logger.info("Initializing GUI")
        self.setWindowTitle(APP_NAME)
        self.predefinedClasses = predefined_class_file
        self.defaultSaveDir = save_directory
        self.project = ProjectDialog(self)
        # Load setting in the main thread
        self.imageData = None
        self.labelFile = None
        # Save as Pascal voc xml
        self.usingPascalVocFormat = True
        self.usingYoloFormat = False
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        # Whether we need to save or not.
        self.dirty = False
        self._noSelectionSlot = False
        self._beginner = True
        # Load predefined classes to the list
        self.loadPredefinedClasses()

        # Main widgets and related state.
        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(
            self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.colorDialog = ColorDialog(parent=self)

        scroll = QScrollArea()
        scroll.setAutoFillBackground(True)
        scroll.setStyleSheet("color:black;")
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = \
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        # noinspection PyTypeChecker
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Group zoom controls into a list for easier toggling.
        # noinspection PyUnresolvedReferences

        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        self.statusBar().showMessage('%s started.' % APP_NAME)
        self.statusBar().show()


        self.filePath = str(filename)



        # Fix the compatible issue for qt4 and qt5.
        # Convert the QStringList to python list
        if self.settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = self.settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [str(i) for i in recentFileQStringList]
            else:
                self.recentFiles = self.settings.get(SETTING_RECENT_FILES)

        size = self.settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = self.settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = str(self.settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = str(self.settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage(
                '%s started. Annotation will be saved to %s' %
                (APP_NAME, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(self.settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(
            self.settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(
            self.settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(self.settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.advancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if default file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDir()

    # Support Functions #
    def noShapes(self):
        return not self.itemsToShapes

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.setCreateMode, self.actions.setEditMode,
                  self.actions.verifyImg)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    # noinspection PyMethodMayBeStatic
    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()


    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes disabled."""
        self.actions.setEditMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            self.logger.info('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

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

    # Add chris
    def buttonState(self):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)

        difficult = self.difficultButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except KeyError:
            pass
        # Checked and Update
        try:
            # noinspection PyUnboundLocalVariable
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(
                    shape, item.checkState() == Qt.Checked)
        except UnboundLocalError:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delBox.setEnabled(selected)
        self.actions.copySelectedShape.setEnabled(selected)
        self.actions.editLabel.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        try:
            shape.paintLabel = self.displayLabelOption.isChecked()
            item = HashableQListWidgetItem(shape.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setBackground(generateColorByText(shape.label))
            self.itemsToShapes[item] = shape
            self.shapesToItems[shape] = item
            self.labelList.addItem(item)
            for action in self.actions.onShapesPresent:
                action.setEnabled(True)
        except AttributeError:
            pass

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

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
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

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
            self.logger.info('Image:{0} -> Annotation:{1}'.format(
                self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.difficultButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or\
                not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.difficultButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color,
                                             generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.setEditMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

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
        natural_sort(images, key=lambda x: x.lower())
        return images

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % APP_NAME
        filters = 'File (*%s)' % LabelFile.suffix
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
                # Return file path without the extension.
                return os.path.splitext(fullFilePath)[0]
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

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



    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toggleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])

def parse_args(args):
    parser = argparse.ArgumentParser()
    img_dir = Flags().imgdir
    random_img = random.choice([os.path.join(img_dir, f) for f in
                                os.listdir(img_dir) if
                                os.path.isfile(os.path.join(img_dir, f))])
    parser.add_argument('-i', '--filename', default=random_img, help="image file to open")
    parser.add_argument('-c', '--predefined_class_file', default=Flags().labels,
                        help="text file containing class names")
    parser.add_argument('-s', '--save_directory', default=None, help="save directory")
    return parser.parse_args(args)


def get_main_app(argv=None):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_()
    -- so that we can test the application in one thread
    """
    if argv is None:
        argv = []
    app = QApplication(argv)
    args = parse_args(sys.argv[1:])

    try:
        # noinspection PyUnresolvedReferences
        import qdarkstyle
        os.environ["QT_API"] = 'pyqt5'
        # detect system theme on macOS
        if sys.platform == "darwin":
            # noinspection PyUnresolvedReferences
            from Foundation import NSUserDefaults as NS
            m = NS.standardUserDefaults().stringForKey_('AppleInterfaceStyle')
            if m == 'Dark':
                # noinspection PyDeprecation
                app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            else:
                pass
        else:
            # noinspection PyDeprecation
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ImportError as e:
        print(" ".join([str(e), "falling back to system theme"]))
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(newIcon("app"))
    win = MainWindow(**vars(args))
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
