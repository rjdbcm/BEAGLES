from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .labelFile import LabelFile
from .stringBundle import StringBundle
from .utils.flags import Flags, FlagIO
from .project import ProjectDialog
#from .scripts.genConfig import genConfigYOLOv2
from absl import logging
import numpy as np
import subprocess
import cv2
import sys
import os
import re
import time


_FLOAT_RE = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')


class Connection(QObject):
    progressUpdate = pyqtSignal(int)


class FloatValidator(QValidator):

    @staticmethod
    def valid_float_string(string):
        match = _FLOAT_RE.search(string)
        return match.groups()[0] == string if match else False

    def validate(self, string, position):
        if self.valid_float_string(string):
            state = self.Acceptable
        elif string == "" or string[position-1] in 'e.-+':
            state = self.Intermediate
        else:
            state = self.Invalid
        return state, string, position

    def fixup(self, text):
        match = _FLOAT_RE.search(text)
        return match.groups()[0] if match else ""


class ScientificDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(float(sys.float_info.min))
        self.setMaximum(float(sys.float_info.max))
        self.validator = FloatValidator()
        self.setDecimals(1000)

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        return self.format_float(value)

    def stepBy(self, steps):
        text = self.cleanText()
        groups = _FLOAT_RE.search(text).groups()
        decimal = float(groups[1])
        decimal += steps
        new_string = "{:g}".format(decimal) + (groups[3] if groups[3] else "")
        self.lineEdit().setText(new_string)

    @staticmethod
    def format_float(value):
        """Modified form of the 'g' format specifier."""
        string = "{:g}".format(value).replace("e+", "e")
        string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
        return string


class FlowThread(QThread, FlagIO):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, proc, flags, rate=.2):
        super(FlowThread, self).__init__(parent)
        self.connection = Connection()
        self.rate = rate
        self.proc = proc
        self.flags = flags
        self.send_flags()

    def stop(self):
        if not self.flags.done:
            self.flags.kill = True
        self.io_flags()
        self.proc.terminate()
        if os.stat(self.logfile.baseFilename).st_size > 0:
            self.logfile.doRollover()
        if os.stat(self.tf_logfile.baseFilename).st_size > 0:
            self.tf_logfile.doRollover()
        self.cleanup_ramdisk()

    def run(self):
        prg = 0
        while self.proc.poll() is None:
            prg_old, prg = prg, self.flags.progress
            if prg > prg_old:
                self.connection.progressUpdate.emit(prg)
            time.sleep(self.rate)
            self.read_flags()


class FlowDialog(QDialog):

    def __init__(self, parent=None, labelfile=None):
        super(FlowDialog, self).__init__(parent)
        self.flags = Flags()
        self.project = ProjectDialog(self)
        self.project.accepted.connect(self.set_project_name)
        self.oldBatchValue = int(self.flags.batch)
        self.oldSaveValue = int(self.flags.save)
        # allow use of labels file passed by slgrSuite

        self.stringBundle = StringBundle.getBundle()

        def getStr(strId):
            return self.stringBundle.getString(strId)

        self.labelfile = labelfile
        self.formGroupBox = QGroupBox("Select Model and Checkpoint")
        layout = QFormLayout()

        self.projectLayout = QHBoxLayout()
        self.projectLbl = QLabel(self.project.default)
        self.projectBtn = QPushButton(getStr('selectProject'))
        self.projectBtn.clicked.connect(self.project.open)
        self.projectLayout.addWidget(self.projectLbl)
        self.projectLayout.addWidget(self.projectBtn)
        layout.addRow(QLabel(getStr('projectName')), self.projectLayout)

        self.flowCmb = QComboBox()
        self.flowCmb.addItems(
            [getStr('train'),
             getStr('predict'),
             getStr('annotate'),
             getStr('analyze')])
        self.flowCmb.setMinimumWidth(100)
        self.flowCmb.currentIndexChanged.connect(self.flowSelect)
        layout.addRow(QLabel("Mode"), self.flowCmb)

        self.modelCmb = QComboBox()
        self.modelCmb.addItems(self.listFiles(self.flags.config))
        self.modelCmb.setToolTip("Choose a model configuration")
        self.modelCmb.currentIndexChanged.connect(self.findCkpt)
        layout.addRow(QLabel("Model"), self.modelCmb)

        self.loadCmb = QComboBox()
        self.loadCmb.setToolTip("Choose a checkpoint")
        self.loadCmb.setMinimumWidth(100)
        layout.addRow(QLabel("Checkpoint"), self.loadCmb)

        self.thresholdSpd = QDoubleSpinBox()
        self.thresholdSpd.setRange(0.0, .99)
        self.thresholdSpd.setSingleStep(0.01)
        self.thresholdSpd.setValue(self.flags.threshold)
        self.thresholdSpd.setDisabled(True)
        layout.addRow(QLabel("Threshold"), self.thresholdSpd)

        self.verbaliseChb = QCheckBox()
        layout.addRow(QLabel("Verbose"), self.verbaliseChb)

        self.formGroupBox.setLayout(layout)

        self.flowGroupBox = QGroupBox("Select Predict Parameters")

        layout2 = QFormLayout()

        self.jsonChb = QCheckBox()
        self.jsonChb.setChecked(False)

        self.vocChb = QCheckBox()
        self.vocChb.setChecked(True)

        layout2.addRow(QLabel("Output JSON Annotations"), self.jsonChb)
        layout2.addRow(QLabel("Output VOC Annotations "), self.vocChb)

        self.flowGroupBox.setLayout(layout2)
        self.flowGroupBox.hide()

        self.trainGroupBox = QGroupBox("Select Training Parameters")

        layout3 = QFormLayout()

        self.trainerCmb = QComboBox()
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad",
                                  "adagradDA", "momentum", "nesterov", "adam",
                                  "AMSGrad", "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        layout3.addRow(QLabel("Training Algorithm"), self.trainerCmb)

        self.momentumSpd = QDoubleSpinBox()
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip(getStr('momentumTip'))
        layout3.addRow(QLabel("Momentum"), self.momentumSpd)

        self.learningModeCmb = QComboBox()
        self.learningModeCmb.addItems(["triangular", "triangular2",
                                       "exp_range"])
        self.learningModeCmb.setItemData(0,
                                         getStr('triangularTip'),
                                         Qt.ToolTipRole)
        self.learningModeCmb.setItemData(1,
                                         getStr('triangular2Tip'),
                                         Qt.ToolTipRole)
        self.learningModeCmb.setItemData(2,
                                         getStr('expRangeTip'),
                                         Qt.ToolTipRole)
        layout3.addRow(QLabel("Learning Mode"), self.learningModeCmb)

        self.learningRateSpd = ScientificDoubleSpinBox()
        self.learningRateSpd.setValue(self.flags.lr)
        layout3.addRow(QLabel("Initial Learning Rate"), self.learningRateSpd)

        self.maxLearningRateSpd = ScientificDoubleSpinBox()
        self.maxLearningRateSpd.setValue(self.flags.max_lr)
        layout3.addRow(QLabel("Maximum Learning Rate"), self.maxLearningRateSpd)

        self.stepSizeCoefficient = QSpinBox()
        self.stepSizeCoefficient.setRange(2, 10)
        self.stepSizeCoefficient.setValue(self.flags.step_size_coefficient)
        layout3.addRow(QLabel("Step Size Coefficient"),
                       self.stepSizeCoefficient)

        self.keepSpb = QSpinBox()
        self.keepSpb.setValue(self.flags.keep)
        self.keepSpb.setRange(1, 256)
        layout3.addRow(QLabel("Checkpoints to Keep"), self.keepSpb)

        self.batchSpb = QSpinBox()
        self.batchSpb.setRange(1, 256)
        self.batchSpb.setValue(int(self.flags.batch))
        self.batchSpb.setWrapping(True)
        layout3.addRow(QLabel("Batch Size"), self.batchSpb)

        self.epochSpb = QSpinBox()
        self.epochSpb.setRange(1, 65536)
        self.epochSpb.setValue(int(self.flags.epoch))
        layout3.addRow(QLabel("Epochs to Run"), self.epochSpb)

        self.saveSpb = QSpinBox()
        self.saveSpb.setRange(1, 65536)
        self.saveSpb.setValue(self.flags.save)
        self.saveSpb.setWrapping(True)
        layout3.addRow(QLabel("Save Every"), self.saveSpb)

        self.clipLayout = QHBoxLayout()
        self.clipNorm = QSpinBox()
        self.clipNorm.setValue(5)
        self.clipNorm.setDisabled(True)
        self.clipChb = QCheckBox()
        self.clipChb.clicked.connect(self.clipNorm.setEnabled)
        self.clipLayout.addWidget(self.clipChb)
        self.clipLayout.addWidget(QLabel("Norm:"))
        self.clipLayout.addWidget(self.clipNorm)
        layout3.addRow(QLabel("Clip Gradients"), self.clipLayout)

        self.updateAnchorChb = QCheckBox()
        layout3.addRow(QLabel("Update Anchors"), self.updateAnchorChb)

        self.trainGroupBox.setLayout(layout3)

        self.flowPrg = QProgressBar()
        self.flowPrg.setRange(0, 100)
        self.buttonRun = QPushButton("Run")
        self.buttonCancel = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonStop = QPushButton("Stop")
        self.buttonStop.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        self.buttonStop.hide()
        self.buttonRun.clicked.connect(self.accept)
        self.buttonStop.clicked.connect(self.closeEvent)
        self.buttonCancel.rejected.connect(self.close)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.flowGroupBox, 1, 0)
        main_layout.addWidget(self.demoGroupBox, 2, 0)
        main_layout.addWidget(self.trainGroupBox, 3, 0)
        main_layout.setSizeConstraint(QLayout.SetFixedSize)
        main_layout.addWidget(self.buttonRun, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonStop, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.flowPrg, 4, 0, Qt.AlignCenter)
        self.setLayout(main_layout)

        self.setWindowTitle("SLGR-Suite - Machine Learning Tool")
        if self.project.check_open_project():
            self.project.name = self.project.check_open_project()
            self.project.show_classes()
            # swapButtons method fails here
            self.project.buttonLoad.hide()
            self.project.buttonOk.show()
            self.set_project_name()
        self.findCkpt()

    def selectProject(self):
        self.project.exec_()

    def findCkpt(self):
        self.loadCmb.clear()
        checkpoints = self.listFiles(self.flags.backup)
        _model = os.path.splitext(self.modelCmb.currentText())
        l = ['0']
        # a dash followed by a number or numbers followed by a dot
        _regex = re.compile("\-[0-9]+\.")
        for f in checkpoints:
            if f[:len(_model[0])] == _model[0]:
                _ckpt = re.search(_regex, f)
                start, end = _ckpt.span()
                n = f[start + 1:end - 1]
                l.append(n)
                self.buttonRun.setDisabled(False)
            # else:
            #     self.buttonOk.setDisabled(True)
        l = list(map(int, l))
        l.sort(reverse=True)
        l = list(map(str, l))
        self.loadCmb.addItems(l)

    def updateCkptFile(self):
        """write selected checkpoint and model information to checkpoint"""
        regex = re.compile('".*?"')
        model_name = os.path.splitext(self.modelCmb.currentText())[0]
        replacement = "-".join([model_name, self.loadCmb.currentText()])
        file = (os.path.join(self.flags.backup, 'checkpoint'))
        open(file, 'a').close()  # touch
        fh = open(file, 'r')
        data = fh.read()
        fh.close()
        result = regex.sub('"{}"'.format(replacement), data)
        fh = open(file, 'w')
        fh.write(result)
        fh.close()

    def trainerSelect(self):
        self.momentumSpd.setDisabled(True)
        for trainer in ("rmsprop", "momentum", "nesterov"):
            if self.trainerCmb.currentText() == trainer:
                self.momentumSpd.setDisabled(False)

    def flowSelect(self):
        if self.flowCmb.currentText() == "Predict":
            self.flowGroupBox.show()
        else:
            self.flowGroupBox.hide()

        if self.flowCmb.currentText() == "Train":
            self.trainGroupBox.show()
            self.thresholdSpd.setDisabled(True)
        else:
            self.trainGroupBox.hide()
            self.loadCmb.setCurrentIndex(0)
    #
    # def updateAnchors(self):
    #     pass
    #     genConfigYOLOv2()

    def set_project_name(self):
        self.projectLbl.setText(self.project.name)

    def assign_flags(self):
        self.flags.project_name = self.projectLbl.text()
        self.flags.model = os.path.join(
            self.flags.config, self.modelCmb.currentText())
        try:
            self.flags.load = int(self.loadCmb.currentText())
        except ValueError:
            self.flags.load = 0
            pass
        self.flags.trainer = self.trainerCmb.currentText()
        self.flags.grayscale = self.grayscaleChb.checkState()
        self.flags.threshold = self.thresholdSpd.value()
        self.flags.clip = bool(self.clipChb.checkState())
        self.flags.clip_norm = self.clipNorm.value()
        self.flags.clr_mode = self.learningModeCmb.currentText()
        self.flags.verbalise = bool(self.verbaliseChb.checkState())
        self.flags.momentum = self.momentumSpd.value()
        self.flags.lr = self.learningRateSpd.value()
        self.flags.max_lr = self.maxLearningRateSpd.value()
        self.flags.keep = self.keepSpb.value()
        self.flags.batch = self.batchSpb.value()
        self.flags.save = self.saveSpb.value()
        self.flags.epoch = self.epochSpb.value()
        self.flags.labels = self.labelfile  # use labelfile set by slgrSuite
        if self.jsonChb.isChecked():
            self.flags.output_type.append("json")
        if self.vocChb.isChecked():
            self.flags.output_type.append("voc")

    def accept(self):
        """set flags for darkflow and prevent startup if errors anticipated"""
        self.updateCkptFile()  # Make sure TFNet gets the correct checkpoint
        self.flags.get_defaults()  # Reset self.flags
        self.assign_flags()

        if not self.flowCmb.currentText() == "Train" and self.flags.load == 0:
            QMessageBox.warning(self, 'Error', "Invalid checkpoint",
                                QMessageBox.Ok)
            return
        if self.flowCmb.currentText() == "Predict":
            self.flowGroupBox.setDisabled(True)
            options = QFileDialog.Options()
            options = QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            dirname = QFileDialog.getExistingDirectory(self,
                                                       'SLGR-Suite Predict - '
                                                       'Choose Image Folder',
                                                       os.getcwd(),
                                                       options)
            self.flags.imgdir = dirname
            pass
        if self.flowCmb.currentText() == "Train":
            if not self.flags.save % self.flags.batch == 0:
                QMessageBox.warning(self, 'Error',
                                    "The value of 'Save Every' should be "
                                    "divisible by the value of 'Batch Size'",
                                    QMessageBox.Ok)
                return
            dataset = [f for f in os.listdir(self.flags.dataset)
                       if not f.startswith('.')]
            if not dataset:
                QMessageBox.warning(self, 'Error',
                                    'No frames or annotations found',
                                    QMessageBox.Ok)
                return
            else:
                self.flags.train = True
        if self.flowCmb.currentText() == "Annotate":
            formats = ['*.avi', '*.mp4', '*.wmv', '*.mkv', '*.mpeg']
            filters = "Video Files (%s)" % ' '.join(
                formats + ['*%s' % LabelFile.suffix])
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename = QFileDialog.getOpenFileName(self,
                                                   'SLGR-Suite Annotate - '
                                                   'Choose Video file',
                                                   os.getcwd(),
                                                   filters, options=options)
            self.flags.fbf = filename[0]
        if [self.flowCmb.currentText() == "Train"]:
            # create backend subprocess
            proc = subprocess.Popen([sys.executable, os.path.join(
                os.getcwd(), "libs/scripts/wrapper.py")],
                                    stdout=subprocess.PIPE, shell=False)
            self.flowthread = FlowThread(self, proc=proc, flags=self.flags)
            self.flowthread.setTerminationEnabled(True)
            self.flowthread.finished.connect(self.onFinished)
            self.flowthread.connection.progressUpdate.connect(
                self.updateProgress)
            self.flowthread.start()
        self.flowPrg.setMaximum(0)
        self.buttonRun.setEnabled(False)
        self.buttonRun.hide()
        self.buttonStop.show()
        self.formGroupBox.setEnabled(False)
        self.trainGroupBox.setEnabled(False)

    def closeEvent(self, event):

        def acceptEvent():
            self.buttonRun.setDisabled(False)
            self.buttonStop.hide()
            self.buttonRun.show()
            self.flowGroupBox.setEnabled(True)
            self.demoGroupBox.setEnabled(True)
            self.trainGroupBox.setEnabled(True)
            self.formGroupBox.setEnabled(True)
            # self.findProject()
            try:
                event.accept()
            except AttributeError:
                pass

        try:
            thread_running = self.flowthread.isRunning()
        except AttributeError:
            thread_running = False
        if thread_running:
            option = "close" if type(event) == QCloseEvent else "stop"
            msg = "Are you sure you want to {} this dialog? " \
                  "This will terminate any running processes.".format(option)
            reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Yes,
                                         QMessageBox.No)
            if reply == QMessageBox.No:
                try:
                    event.ignore()
                except AttributeError:
                    pass
            else:
                try:
                    self.flowthread.stop()
                except AttributeError:
                    pass
                acceptEvent()

        else:
            self.flowPrg.setMaximum(100)
            self.flowPrg.reset()
            acceptEvent()

    def onFinished(self):
        self.flags = self.flowthread.flags
        if self.flags.error:
            QMessageBox.critical(self, "Error Message", self.flags.error,
                                 QMessageBox.Ok)
            if os.stat(self.flowthread.logfile.baseFilename).st_size > 0:
                self.flowthread.logfile.doRollover()
            if os.stat(self.flowthread.tf_logfile.baseFilename).st_size > 0:
                self.flowthread.tf_logfile.doRollover()
        if self.flags.verbalise:
            QMessageBox.information(self, "Debug Message", "Process Stopped:\n"
                                    + "\n".join('{}: {}'.format(k, v)
                                                for k, v in
                                                self.flags.items()),
                                    QMessageBox.Ok)
        self.flowGroupBox.setEnabled(True)
        self.demoGroupBox.setEnabled(True)
        self.trainGroupBox.setEnabled(True)
        self.formGroupBox.setEnabled(True)
        self.flowPrg.setMaximum(100)
        self.flowPrg.reset()
        self.buttonRun.setDisabled(False)
        self.buttonStop.hide()
        self.buttonRun.show()
        self.findCkpt()
        # self.findProject()

    @pyqtSlot(int)
    def updateProgress(self, value):
        if self.flowPrg.maximum():
            self.flowPrg.setValue(value)
        else:  # stop pulsing and set value
            self.flowPrg.setMaximum(100)
            self.flowPrg.setValue(value)

    # HELPERS
    @staticmethod
    def listFiles(path):
        path = QDir(path)
        filters = ["*.cfg", "*.index"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
