from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .stringBundle import StringBundle

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
