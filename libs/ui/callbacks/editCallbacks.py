from libs.io.labelFile import LabelFile
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.ui.functions.editFunctions import EditFunctions


class EditCallbacks(EditFunctions):
    def editLabel(self):
        if not self.canvas.editing:
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(self.generateColorByText(text))
            self.setDirty()

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def createShape(self):
        assert self.beginner()
        self.canvas.editing = False
        # noinspection PyUnresolvedReferences
        self.actions.create.setEnabled(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def hideAll(self):
        self.togglePolygons(False)

    def showAll(self):
        self.togglePolygons(True)

    def delBox(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            # noinspection PyUnresolvedReferences
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def shapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.updates_data()
            self.setDirty()

    def shapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.updates_data()
            self.setDirty()

    def boxLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.updates_data()
            self.setDirty()

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def verifyImg(self, _value=False):
        # Proceeding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.labelFile = LabelFile()
                self.logger.info(self.labelFile)
                if self.labelFile is not None:
                    self.labelFile.toggleVerify()
                else:
                    return
            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()