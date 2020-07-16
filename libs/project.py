import os
import json
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
        self.dirty = False
        self.default = Flags().project_name
        self.projects = next(os.walk(Flags().summary))[1]

        layout = QFormLayout()
        self.formGroupBox = QGroupBox()
        self.projectHBox = QHBoxLayout()
        self.projectCmb = QComboBox()
        self.projectCmb.addItems(self.projects)
        self.projectCmb.setEditable(True)
        self.projectCmb.currentTextChanged.connect(self._change_name)
        layout.addRow("Project Name", self.projectCmb)
        self.projectClasses = QTextEdit()
        self.projectCmb.currentIndexChanged.connect(self.disable_class_list)
        layout.addRow("Class List", self.projectClasses)
        self.formGroupBox.setLayout(layout)

        self.buttonCancel = QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.reject)
        self.buttonSave = QPushButton("Save to Archive")
        self.buttonSave.clicked.connect(self.save)
        self.buttonLoad = QPushButton("Load Project")
        self.buttonLoad.clicked.connect(self.load)
        self.buttonOk = QPushButton("Ok")
        self.buttonOk.hide()
        self.buttonOk.clicked.connect(self.accept)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.buttonSave, 4, 0, Qt.AlignCenter)
        main_layout.addWidget(self.buttonOk, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonLoad, 4, 0, Qt.AlignRight)

        self.setLayout(main_layout)
        self.name = "default"
        self.setWindowTitle("SLGR-Suite - Load a Project")

    def disable_class_list(self):
        self.projectClasses.setDisabled(True)
        self.projectClasses.setText("Load a Project to edit it's class list")

    def _change_name(self):
        self.name = self.projectCmb.currentText()

    def show_classes(self):
        classes = self.read_class_list()
        self.projectClasses.setText(classes)

    @staticmethod
    def read_class_list():
        l = list()
        with open(Flags().labels) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                l.append(line)
        return ''.join(l)

    def write_class_list(self):
        data = self.projectClasses.toPlainText()
        if len(data) != len(Flags().labels):
            file = open(Flags().labels, "w")
            file.write(data)
            file.close()

    def load(self):
        if self.dirty:
            QMessageBox.warning(self, 'Project Load Error',
                                'Save changes before loading a new project')
            return
        self.clear_sandbox()
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
                self.archive()
        self.show_classes()
        self.projectClasses.setEnabled(True)
        self.swapButtons()
        self.dirty = True

    def swapButtons(self):
        if self.buttonLoad.isVisible() and self.buttonOk.isHidden():
            self.buttonLoad.hide()
            self.buttonOk.show()
        elif self.buttonLoad.isHidden() and self.buttonOk.isVisible():
            self.buttonLoad.show()
            self.buttonOk.hide()

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
        self.swapButtons()
        self.dirty = False
