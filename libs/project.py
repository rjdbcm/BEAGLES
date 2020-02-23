import os
import shutil
import tarfile
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .utils.flags import Flags


class projectSignal(QObject):
    saveFilename = pyqtSignal(str)
    loadFilename = pyqtSignal(str)


class projectDialog(QDialog):
    def __init__(self, parent):
        super(projectDialog, self).__init__(parent)
        self.signals = projectSignal()
        self.signals.loadFilename.connect(self.load)
        self.signals.saveFilename.connect(self.save)
        self.default = os.path.join(Flags().summary, "default", "default.tar")
        self.projects = next(os.walk(Flags().summary))[1]

        layout = QFormLayout()
        self.formGroupBox = QGroupBox()
        self.projectCmb = QComboBox()
        self.projectCmb.addItems(self.projects)
        layout.addRow("Project Name", self.projectCmb)
        self.projectClasses = QTextEdit()
        layout.addRow("Class List", self.projectClasses)
        self.formGroupBox.setLayout(layout)

        self.buttonOk = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonOk.clicked.connect(self.accept)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.buttonOk, 4, 0, Qt.AlignRight)

        self.setWindowTitle("SLGR-Suite - Load a Project")

    @pyqtSlot(str)
    def load(self, file):
        cond = True
        while cond:
            try:
                with tarfile.TarFile(file, 'r', errorlevel=1) as archive:
                    for i in archive.getnames():
                        try:
                            archive.extract(i)
                        except OSError:
                            os.remove(i)
                            archive.extract(i)
                    cond = False
            except FileNotFoundError:
                os.mkdir(os.path.dirname(file))
                shutil.copy(os.path.join(Flags().summary,
                                         Flags().project_name,
                                         Flags().project_name + ".tar"), file)

    # def restore(self):

    @pyqtSlot(str)
    def save(self, name):
        archiveList = [Flags().binary,
                       Flags().built_graph,
                       Flags().backup,
                       Flags().dataset,
                       Flags().video_out,
                       Flags().img_out,
                       './data/rawframes/',
                       self.predefinedClasses]
        archive = os.path.join(Flags().summary, name,
                               name + '.tar')
        with tarfile.open(archive, mode='w') as archive:
            for i in archiveList:
                archive.add(i)
                try:
                    shutil.rmtree(i)
                except NotADirectoryError:
                    os.remove(i)

        # reload default dirs
        self.signals.loadFilename.emit(self.default)

    def accept(self):
        name, ext = [self.projectCmb.currentText(), ".tar"]
        projectFile = os.path.join(Flags().summary, name, name + ext)
        self.signals.loadFilename.emit(projectFile)

    def closeEvent(self, event):
        msg = "Are you sure you want to close the project selection dialog?"
        reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.No:
            try:
                event.ignore()
            except AttributeError:
                pass
        else:
            event.accept()
