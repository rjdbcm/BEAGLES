import os
import json
from libs.zoomWidget import ZoomWidget
from libs.settings import Settings
from libs.canvas import Canvas
from libs.constants import *
from libs.ui.callbacks.actionCallbacks import ActionCallbacks
from functools import partial
from libs.stringBundle import getStr
from libs.toolBar import ToolBar
from libs.qtUtils import *
from libs.utils.flags import FlagIO


class BeaglesMainWindow(QMainWindow, ActionCallbacks, FlagIO):
    def __init__(self):
        super(BeaglesMainWindow, self).__init__()
        FlagIO.__init__(self, subprogram=True)
        with open('resources/actions/actions.json', 'r') as json_file:
            self.actionSettings = json.load(json_file)
        self.actionList = list(self.actionSettings.keys())
        self._beginner = True
        action = partial(newAction, self)

        def createActions(actions: list):
            nonlocal self
            nonlocal action
            cmd = 'global {0}; {0} = action("{1}", {2}, "{3}", "{4}", "{5}", {6}, {7})'
            for act in actions:
                _str = act
                action_str = getStr(_str)
                action_shortcut, checkable, enabled = [str(i) for i in self.actionSettings[_str]]
                action_detail = getStr(_str + "Detail")
                action_icon = _str
                callback = 'self.' + act
                cmd_string = cmd.format(_str, action_str, callback, action_shortcut,
                                        action_icon, action_detail, checkable, enabled)
                self.logger.info(cmd_string)
                exec(cmd_string)
        createActions(self.actionList)
        self.setup()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        toolbar.setMovable(False)
        toolbar.setFixedHeight(32)
        toolbar.setIconSize(QSize(30, 30))
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    def setup(self):
        self.settings = Settings()
        self.settings.load()
        self.setupZoomWidget()
        self.setupActions()
        self.setupMenus()
        self.setupToolbar()
        self.setupFileDock()
        self.setupLabelDock()
        self.populateMenus()
        self.setupCanvasWidget()
        self.populateModeActions()

    def setupActions(self):
        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(self.settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(self.settings.get(SETTING_SINGLE_CLASS, False))
        # Add option to enable/disable labels being displayed at the top of bounding boxes

        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(self.settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(self.settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toggleDrawSquare)
        # noinspection PyUnresolvedReferences
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg, setFitWin, setFitWidth)
        # noinspection PyUnresolvedReferences
        self.actions = Struct(saveFile=saveFile, changeFormat=changeFormat, saveAs=saveAs,
                              openFile=openFile, closeFile=closeFile, resetAll=resetAll,
                              verifyImg=verifyImg, lineColor=boxLineColor,
                              create=createShape, delBox=delBox, editLabel=editLabel,
                              copySelectedShape=copySelectedShape, trainModel=trainModel,
                              visualize=visualize, setCreateMode=setCreateMode,
                              setEditMode=setEditMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=self.zoom, zoomIn=zoomIn, zoomOut=zoomOut,
                              zoomOrg=zoomOrg, setFitWin=setFitWin, setFitWidth=setFitWidth,
                              zoomActions=zoomActions, fileMenuActions=(openFile, openDir,
                              impVideo, saveFile, saveAs, commitAnnotatedFrames, trainModel,
                              visualize, closeFile, resetAll, close), beginner=(), advanced=(),
                              editMenu=(editLabel, copySelectedShape, delBox, None,
                                        boxLineColor, self.drawSquaresOption),
                              beginnerContext=(createShape, editLabel, copySelectedShape,
                                               delBox),
                              advancedContext=(setCreateMode, setEditMode, editLabel,
                                               copySelectedShape, delBox, shapeLineColor,
                                               shapeFillColor),
                              onLoadActive=(closeFile, createShape, setCreateMode,
                                            setEditMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

    # noinspection PyUnresolvedReferences
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

    def setupMenus(self):
        labelMenu = QMenu()
        # noinspection PyUnresolvedReferences
        addActions(labelMenu, (editLabel, delBox))
        self.menus = Struct(file=self.menu('&File'), edit=self.menu('&Edit'),
                            view=self.menu('&View'),
                            machine_learning=self.menu('&Machine Learning'),
                            help=self.menu('&Help'),
                            recentFiles=QMenu('Open &Recent'), labelList=labelMenu)
        # noinspection PyUnresolvedAttribute
        self.menus.file.aboutToShow.connect(self.updateFileMenu)

    def populateMenus(self):
        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')
        # noinspection PyUnresolvedReferences
        addActions(self.menus.file, (openFile, openDir, changeSaveDir, impVideo,
                                     openAnnotation, self.menus.recentFiles, saveFile,
                                     changeFormat, saveAs, closeFile, resetAll, close))
        # noinspection PyUnresolvedReferences
        addActions(self.menus.help, (showTutorialDialog, showInfo))
        # noinspection PyUnresolvedReferences
        addActions(self.menus.view, (self.autoSaving, self.singleClassMode,
                                     self.displayLabelOption, labels, advancedMode, None,
                                     prevImg, nextImg, None, hideAll, showAll, None,
                                     zoomIn, zoomOut, zoomOrg, None, setFitWin,
                                     setFitWidth))
        # noinspection PyUnresolvedReferences
        addActions(self.menus.machine_learning, (commitAnnotatedFrames, trainModel,
                                                 visualize, None))

    def setupToolbar(self):
        self.tools = self.toolbar('Tools')
        # noinspection PyUnresolvedReferences
        self.actions.beginner = (openFile, openDir, changeSaveDir, saveFile, None,
                                 createShape, copySelectedShape, delBox, None, prevImg,
                                 nextImg, None, zoomIn, self.zoom, zoomOut, setFitWin,
                                 setFitWidth)

        # noinspection PyUnresolvedReferences
        self.actions.advanced = (openFile, saveFile, None, setCreateMode, setEditMode,
                                 verifyImg, None, hideAll, showAll, None,
                                 commitAnnotatedFrames, trainModel, visualize, impVideo)

    def setupFileDock(self):
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

    def setupLabelDock(self):
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # noinspection PyUnresolvedReferences
        self.editButton.setDefaultAction(editLabel)
        # Create a widget for edit and diffc button
        self.difficultButton = QCheckBox(getStr('useDifficult'))
        self.difficultButton.setChecked(False)
        self.difficultButton.stateChanged.connect(self.buttonState)
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)
        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dockFeatures = \
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        # noinspection PyTypeChecker
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)
        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.difficultButton)
        listLayout.addWidget(useDefaultLabelContainer)
        self.lastLabel = None

    def setupZoomWidget(self):
        self.zoom = QWidgetAction(self)
        self.zoomWidget = ZoomWidget()
        self.zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

    def setupCanvasWidget(self):
        action = partial(newAction, self)
        # Create and add a widget for showing current label items
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(self.settings.get(SETTING_DRAW_SQUARE, False))
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        # Custom context menu for the canvas widget:
        # noinspection PyUnresolvedReferences
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (action('&Copy here', self.copyShape),
                                          action('&Move here', self.moveShape)))

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(BeaglesMainWindow, self).resizeEvent(event)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        if self.tb_process.pid() > 0:
            self.tb_process.kill()
        settings = self.settings
        # If it loads images from dir, don't load it at the beginning
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''
        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner

        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = str(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings.save()
