#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import os.path
import argparse
from functools import partial

# Add internal libs
# noinspection PyUnresolvedReferences
from libs.resources import *
from libs.widgets.beaglesMainWindow import BeaglesMainWindow
from libs.constants import *
from libs.qtUtils import *
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.widgets.labelDialog import LabelDialog
from libs.utils.flags import Flags
from libs.widgets.colorDialog import ColorDialog
from libs.widgets.projectDialog import ProjectDialog
from libs.widgets.hashableQListWidgetItem import HashableQListWidgetItem


class MainWindow(BeaglesMainWindow):

    # noinspection PyShadowingBuiltins
    def __init__(self, filename=None, predefined_class_file=None,
                 save_directory=None, darkmode=None):
        super(MainWindow, self).__init__()
        self.logger.info("Initializing GUI")
        self.setWindowTitle(APP_NAME)
        self.predefinedClasses = predefined_class_file
        self.defaultSaveDir = save_directory
        self.darkmode = darkmode
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

        # Load predefined classes to the list
        self.loadPredefinedClasses()

        # Main widgets and related state.
        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

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
        self.setCentralWidget(scroll)
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
            if haveQString():
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
            return x.toBool() if isinstance(x, QVariant) else bool(x)

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


        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if default file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDir()

    # Support Functions #
    def noShapes(self):
        return not self.itemsToShapes

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes disabled."""
        self.actions.setEditMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            self.logger.info('Cancel creation.')
            self.canvas.editing = True
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    # Add chris
    def buttonState(self):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing:
            return

        # If not selected Item, take the first one
        item = self.currentItem()
        item = self.labelList.item(self.labelList.count() - 1) if not item else item

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
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
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

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing:
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
                self.canvas.editing = True
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

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()


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
    parser.add_argument('-d', '--darkmode', default=True,
                        help='use qdarkstyle (defaults to system theme on macos)')
    return parser.parse_args(args)


def get_main_app(argv=None):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_()
    -- so that we can test the application in one thread
    """
    argv = [] if argv is None else lambda: None
    app = QApplication(argv)
    args = parse_args(sys.argv[1:])
    darkmode(app) if args.darkmode else lambda: None
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(newIcon("app"))
    win = MainWindow(**vars(args))
    win.show()
    return app, win


def darkmode(app):
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


def main():
    """construct main app and run it"""
    app, _win = get_main_app()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
