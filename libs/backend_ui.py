import re
import os
import sys
import time
from libs.utils.flags import FlagIO
from .stringBundle import StringBundle
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

_FLOAT_RE = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')


class BackendConnection(QObject):
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


class BackendThread(QThread, FlagIO):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, proc, flags, rate=.2):
        super(BackendThread, self).__init__(parent)
        self.connection = BackendConnection()
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


class BackendDialog(QDialog):
    def __init__(self, parent):
        super(BackendDialog, self).__init__(parent)
        self.stringBundle = StringBundle.getBundle()

    def setupDialog(self):
        def getStr(strId):
            return self.stringBundle.getString(strId)

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
        layout3.addRow(QLabel("Maximum Learning Rate"),
                       self.maxLearningRateSpd)

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
        self.buttonStop.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserStop))
        self.buttonStop.hide()
        self.buttonRun.clicked.connect(self.accept)
        self.buttonStop.clicked.connect(self.closeEvent)
        self.buttonCancel.rejected.connect(self.close)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.flowGroupBox, 1, 0)
        main_layout.addWidget(self.trainGroupBox, 3, 0)
        main_layout.setSizeConstraint(QLayout.SetFixedSize)
        main_layout.addWidget(self.buttonRun, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonStop, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.flowPrg, 4, 0, Qt.AlignCenter)
        self.setLayout(main_layout)

        self.setWindowTitle("BEAGLES - Machine Learning Tool")