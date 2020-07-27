from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from functools import partial
from .stringBundle import StringBundle
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
        self.shortcuts = dict({
            'impVideo': 'Ctrl+i',
            'nextImg': 'd',
            'prevImg': 'a'
        })

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

    # def _generateActions(self, actions: list, checkable=False, enabled=True):
    #     new_action = partial(newAction, self)
    #     for action in actions:
    #
    #         print(action_str, action, action_shortcut,
    #                                      action_icon, action_detail, checkable,
    #                                      enabled)
    #         complete_action = new_action(action_str, action, action_shortcut,
    #                                      action_icon, action_detail, checkable,
    #                                      enabled)
    #         yield complete_action
