from beagles.ui.functions.mainWindowFunctions import MainWindowFunctions


class EditFunctions(MainWindowFunctions):
    def toggleDrawMode(self, edit=True):
        self.canvas.editing = True
        # noinspection PyUnresolvedReferences
        self.actions.setCreateMode.setEnabled(edit)
        # noinspection PyUnresolvedReferences
        self.actions.setEditMode.setEnabled(not edit)
