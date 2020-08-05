from libs.ui.functions.viewFunctions import ViewFunctions


class ViewCallbacks(ViewFunctions):
    def zoomIn(self):
        self.addZoom(10)

    def zoomOut(self):
        self.addZoom(-10)

    def zoomOrg(self):
        self.setZoom(100)

    def setFitWin(self, value=True):
        if value:
            # noinspection PyUnresolvedReferences
            self.actions.setFitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            # noinspection PyUnresolvedReferences
            self.actions.setFitWin.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def nextImg(self, _value=False):
        if self.autoSaving.isChecked():
            self.autoSave()
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def prevImg(self, _value=False):
        if self.autoSaving.isChecked():
            self.autoSave()
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def advancedMode(self, value=True):
        self._beginner = not value
        self.canvas.editing = True
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            # noinspection PyUnresolvedReferences
            self.actions.setCreateMode.setEnabled(True)
            # noinspection PyUnresolvedReferences
            self.actions.setEditMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)
