from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from libs.zoomWidget import ZoomWidget
from libs.settings import Settings
from libs.canvas import Canvas
from libs.constants import *
from functools import partial
from .stringBundle import StringBundle
import json
from .toolBar import ToolBar
from .qtUtils import *
from .utils.flags import FlagIO


stringBundle = StringBundle.getBundle()


def getStr(strId):
    return stringBundle.getString(strId)


class BeaglesDialog(QDialog):
    def __init__(self, parent):
        super(BeaglesDialog, self).__init__(parent)

    def getWidgets(self) -> dict:
        return self.__dict__

    def _getWidget(self, index: int) -> dict:
        widgets = self.getWidgets()
        widget_name = list(widgets)[index]
        widget_obj = list(widgets.values())[index]
        return {widget_name: widget_obj}

    def getWidgetsByIndex(self, start, end) -> dict:
        d = dict()
        for i in range(start, end):
            d.update(self._getWidget(i))
        return d

    def addRowsToLayout(self, layout, widgets: dict):
        for key, obj in widgets.items():
            label = QLabel(getStr(str(key)))
            layout.addRow(label, obj)

    # def addWidgetsToLayout(self, layout, widgets: dict):
    #     for _, obj in widgets.items():
    #         layout.addWidget(obj)


class BeaglesMainWindow(QMainWindow, FlagIO):
    def __init__(self):
        super(BeaglesMainWindow, self).__init__()
        FlagIO.__init__(self, subprogram=True)
        with open('resources/actions/actions.json', 'r') as json_file:
            self.actionSettings = json.load(json_file)
        """dict(key: list(shortcut: str, checkable: bool, enabled: bool))"""
        self.actionList = list(self.actionSettings.keys())
        action = partial(newAction, self)

        def createActions(actions: list):
            nonlocal self
            nonlocal action
            cmd = 'global {0}; {0} = action("{1}", {2}, "{3}", "{4}", "{5}", {6}, {7})'
            for act in actions:
                _str = act
                action_str = getStr(_str)
                action_shortcut, checkable, enabled = [str(i) for i in
                                                       self.actionSettings[
                                                           _str]]
                action_detail = getStr(_str + "Detail")
                action_icon = _str
                callback = 'self.' + act
                self.logger.info(
                    cmd.format(_str, action_str, callback, action_shortcut,
                               action_icon, action_detail, checkable,
                               enabled))
                exec(cmd.format(_str, action_str, callback, action_shortcut,
                                action_icon, action_detail, checkable,
                                enabled))
        createActions(self.actionList)
        self.settings = Settings()
        self.settings.load()
        settings = self.settings
        self.zoom = QWidgetAction(self)
        self.zoomWidget = ZoomWidget()
        self.zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, setFitWin, setFitWidth)
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE,
                                                       False))
        self.drawSquaresOption.triggered.connect(self.toggleDrawSquare)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.editButton.setDefaultAction(editLabel)
        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        labelMenu = QMenu()
        # noinspection PyUnresolvedReferences
        addActions(labelMenu, (editLabel, delBox))
        self.actions = Struct(saveFile=saveFile, changeFormat=changeFormat,
                              saveAs=saveAs, openFile=openFile,
                              closeFile=closeFile,
                              resetAll=resetAll, verifyImg=verifyImg,
                              lineColor=boxLineColor, create=createShape,
                              delBox=delBox, editLabel=editLabel,
                              copySelectedShape=copySelectedShape,
                              trainModel=trainModel, visualize=visualize,
                              setCreateMode=setCreateMode,
                              setEditMode=setEditMode,
                              advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor,
                              shapeFillColor=shapeFillColor,
                              zoom=self.zoom, zoomIn=zoomIn, zoomOut=zoomOut,
                              zoomOrg=zoomOrg,
                              setFitWin=setFitWin, setFitWidth=setFitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  openFile, openDir, impVideo, saveFile,
                                  saveAs,
                                  commitAnnotatedFrames, trainModel, visualize,
                                  closeFile, resetAll, close),
                              beginner=(), advanced=(),
                              editMenu=(editLabel, copySelectedShape, delBox,
                                        None, boxLineColor,
                                        self.drawSquaresOption),
                              beginnerContext=(createShape, editLabel, copySelectedShape,
                                               delBox),
                              advancedContext=(setCreateMode, setEditMode,
                                               editLabel, copySelectedShape, delBox,
                                               shapeLineColor, shapeFillColor),
                              onLoadActive=(closeFile, createShape,
                                            setCreateMode, setEditMode),
                              onShapesPresent=(saveAs, hideAll, showAll))
        self.menus = Struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)
        # Create and add a widget for showing current label items
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(
            self.settings.get(SETTING_DRAW_SQUARE,
                              False))
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        # Create a widget for using default label
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
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

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
        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(self.settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(self.settings.get(SETTING_SINGLE_CLASS,
                                                         False))
        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(
            self.settings.get(SETTING_PAINT_LABEL,
                              False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)
        # noinspection PyUnresolvedReferences
        addActions(self.menus.file,
                   (openFile, openDir, changeSaveDir, impVideo, openAnnotation,
                    self.menus.recentFiles, saveFile, changeFormat,
                    saveAs, trainModel, closeFile, resetAll, close))
        # noinspection PyUnresolvedReferences
        addActions(self.menus.help, (showTutorialDialog, showInfo))
        # noinspection PyUnresolvedReferences
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            prevImg, nextImg, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            setFitWin, setFitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')

        # noinspection PyUnresolvedReferences
        self.actions.beginner = (
            openFile, openDir, changeSaveDir, saveFile, None,
            createShape, copySelectedShape, delBox, None, prevImg, nextImg,
            None, zoomIn,
            self.zoom, zoomOut, setFitWin, setFitWidth)

        # noinspection PyUnresolvedReferences
        self.actions.advanced = (
        openFile, saveFile, None, setCreateMode, setEditMode,
        verifyImg, None, hideAll, showAll, None,
        commitAnnotatedFrames, trainModel, visualize,
        impVideo)

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