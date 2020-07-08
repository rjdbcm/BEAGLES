import os
import pickle
import shutil
import tarfile
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from libs.utils.flags import Flags


class ProjectDialog(QDialog):

    def __init__(self, parent):
        super(ProjectDialog, self).__init__(parent)
        self.setModal(True)
        self.archive_list = [Flags().binary,
                             Flags().built_graph,
                             Flags().backup,
                             Flags().dataset,
                             Flags().video_out,
                             Flags().img_out,
                             './data/rawframes/',
                             Flags().labels]
        self.default = "Sandbox Mode"
        self.projects = next(os.walk(Flags().summary))[1]

        layout = QFormLayout()
        self.formGroupBox = QGroupBox()
        self.projectCmb = QComboBox()
        self.projectCmb.addItems(self.projects)
        self.projectCmb.setEditable(True)
        self.projectCmb.currentTextChanged.connect(self._change_name)
        layout.addRow("Project Name", self.projectCmb)
        self.projectClasses = QTextEdit()
        self.projectClasses.setText(self._read_classes())
        layout.addRow("Class List", self.projectClasses)
        self.formGroupBox.setLayout(layout)

        self.buttonCancel = QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.reject)
        self.buttonSave = QPushButton("Save to Archive")
        self.buttonSave.clicked.connect(self.save)
        self.buttonLoad = QPushButton("Load Project")
        self.buttonLoad.clicked.connect(self.load)
        self.buttonLoad.clicked.connect(self.accept)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.buttonSave, 4, 0, Qt.AlignCenter)
        main_layout.addWidget(self.buttonLoad, 4, 0, Qt.AlignRight)

        self.setLayout(main_layout)
        self.name = "default"
        self.setWindowTitle("SLGR-Suite - Load a Project")

    def _change_name(self):
        self.name = self.projectCmb.currentText()

    @staticmethod
    def _read_classes():
        with open(Flags().labels) as classes:
            try:
                data = classes.read()
            except TypeError:
                pass
        return data

    def write_class_list(self):
        data = self.projectClasses.toPlainText()
        if len(data) != len(Flags().labels):
            file = open(Flags().labels, "w")
            file.write(data)
            file.close()

    def load(self):
        file = self.projectCmb.currentText()
        archive = os.path.join(Flags().summary, file,
                            file + ".tar")
        cond = True
        while cond:
            try:
                with tarfile.TarFile(archive, 'r', errorlevel=1) as archive:
                    for i in archive.getnames():
                        try:
                            archive.extract(i)
                        except OSError:
                            os.remove(i)
                            archive.extract(i)
                    cond = False
            except FileNotFoundError:
                os.mkdir(os.path.dirname(archive))
                shutil.copy(archive, file)

    def clear_sandbox(self):
        for i in self.archive_list:
            if os.path.isdir(i):
                [os.remove(j.path) for j in os.scandir(i)]
            elif os.path.isfile(i):
                open(i, "w").close()

    def archive(self):
        name = self.projectCmb.currentText()
        if name != Flags().project_name:
            archive = os.path.join(Flags().summary, name,
                                   name + '.tar')
            while True:
                try:
                    with tarfile.TarFile(archive, mode='w', errorlevel=1) as archive:
                        for i in self.archive_list:
                            archive.add(i)
                        break
                except FileNotFoundError:
                    os.mkdir(os.path.join(Flags().summary, name))

    def save(self):
        self.write_class_list()
        self.archive()