import os
import time
from libs.widgets.scientificQDoubleSpinBox import ScientificQDoubleSpinBox
from libs.stringBundle import getStr
from libs.io.flags import FlagIO
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, \
                            QDialog, QLabel, QPushButton, QGridLayout, QProgressBar, \
                            QDialogButtonBox, QStyle, QLayout, QFormLayout, QGroupBox



class BackendConnection(QObject):
    progressUpdate = pyqtSignal(int)


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
        # training_widgets
        self.flowCmb = QComboBox()
        self.modelCmb = QComboBox()
        self.loadCmb = QComboBox()
        self.thresholdSpd = QDoubleSpinBox()
        self.verbaliseChb = QCheckBox()
        # predict_widgets
        self.jsonChb = QCheckBox()
        self.vocChb = QCheckBox()
        # hyperparameter_widgets
        self.trainerCmb = QComboBox()
        self.momentumSpd = QDoubleSpinBox()
        self.learningModeCmb = QComboBox()
        self.learningRateSpd = ScientificQDoubleSpinBox()
        self.maxLearningRateSpd = ScientificQDoubleSpinBox()
        self.stepSizeCoefficient = QSpinBox()
        self.keepSpb = QSpinBox()
        self.batchSpb = QSpinBox()
        self.epochSpb = QSpinBox()
        self.saveSpb = QSpinBox()
        self.clipLayout = QHBoxLayout()
        self.updateAnchorChb = QCheckBox()

    @staticmethod
    def addRowsToLayout(layout, widgets: dict):
        for key, obj in widgets.items():
            label = QLabel(getStr(str(key)))
            layout.addRow(label, obj)

    @property
    def widgets(self) -> dict:
        return self.__dict__

    def widgetFromIndex(self, index: int) -> dict:
        widget_name = list(self.widgets)[index]
        widget_obj = list(self.widgets.values())[index]
        return {widget_name: widget_obj}

    def getWidgetsByIndex(self, start, end) -> dict:
        d = dict()
        for i in range(start, end):
            d.update(self.widgetFromIndex(i))
        return d

    def setupProjectWidgets(self, layout) -> None:
        self.projectLayout = QHBoxLayout()
        self.projectBtn = QPushButton(getStr('selectProject'))
        self.projectLbl = QLabel(self.project.default)
        self.projectBtn.clicked.connect(self.project.open)
        self.projectLayout.addWidget(self.projectLbl)
        self.projectLayout.addWidget(self.projectBtn)
        layout.addRow(QLabel("Project"), self.projectLayout)

    def setupTrainingWidgets(self):
        self.flowCmb.addItems(
            [getStr('train'),
             getStr('predict'),
             getStr('annotate'),
             getStr('analyze')])
        self.flowCmb.setMinimumWidth(100)
        self.flowCmb.currentIndexChanged.connect(self.flowSelect)
        self.modelCmb.addItems(self.listFiles(self.flags.config))
        self.modelCmb.setToolTip("Choose a model configuration")
        self.modelCmb.currentIndexChanged.connect(self.findCkpt)
        self.loadCmb.setToolTip("Choose a checkpoint")
        self.loadCmb.setMinimumWidth(100)
        self.thresholdSpd.setRange(0.0, .99)
        self.thresholdSpd.setSingleStep(0.01)
        self.thresholdSpd.setValue(self.flags.threshold)
        self.thresholdSpd.setDisabled(True)

    def setupPredictWidgets(self):
        self.jsonChb.setChecked(False)
        self.vocChb.setChecked(True)

    def setupHyperparameterWidgets(self):
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad",
                                  "adagradDA", "momentum", "nesterov", "adam",
                                  "AMSGrad", "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip(getStr('momentumTip'))
        learning_modes = ["triangular", "triangular2", "exp_range"]
        self.learningModeCmb.addItems(learning_modes)
        for i, mode in enumerate(learning_modes):
            self.learningModeCmb.setItemData(i, getStr(mode + 'Tip'),
                                             Qt.ToolTipRole)
        self.learningRateSpd.setValue(self.flags.lr)
        self.maxLearningRateSpd.setValue(self.flags.max_lr)
        self.stepSizeCoefficient.setRange(2, 10)
        self.stepSizeCoefficient.setValue(self.flags.step_size_coefficient)
        self.keepSpb.setValue(self.flags.keep)
        self.keepSpb.setRange(1, 256)
        self.batchSpb.setRange(1, 256)
        self.batchSpb.setValue(int(self.flags.batch))
        self.batchSpb.setWrapping(True)
        self.epochSpb.setRange(1, 65536)
        self.epochSpb.setValue(int(self.flags.epoch))
        self.saveSpb.setRange(1, 65536)
        self.saveSpb.setValue(self.flags.save)
        self.saveSpb.setWrapping(True)
        self.clipNorm = QSpinBox()
        self.clipChb = QCheckBox()
        self.clipNorm.setValue(5)
        self.clipNorm.setDisabled(True)
        self.clipChb.clicked.connect(self.clipNorm.setEnabled)
        self.clipLayout.addWidget(self.clipChb)
        self.clipLayout.addWidget(QLabel("Norm:"))
        self.clipLayout.addWidget(self.clipNorm)

    def setupMainLayout(self):
        self.flowPrg = QProgressBar()
        self.buttonRun = QPushButton("Run")
        self.buttonStop = QPushButton("Stop")
        self.flowPrg.setRange(0, 100)
        self.buttonCancel = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonStop.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserStop))
        self.buttonStop.hide()
        self.buttonRun.clicked.connect(self.accept)
        self.buttonStop.clicked.connect(self.closeEvent)
        self.buttonCancel.rejected.connect(self.close)

        main_layout = QGridLayout()
        main_layout.setSizeConstraint(QLayout.SetFixedSize)
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.flowGroupBox, 1, 0)
        main_layout.addWidget(self.trainGroupBox, 3, 0)
        main_layout.addWidget(self.buttonRun, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonStop, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.flowPrg, 4, 0, Qt.AlignCenter)
        self.setLayout(main_layout)

    def setupDialog(self):
        layout = QFormLayout()
        self.formGroupBox = QGroupBox("Select Model and Checkpoint")
        self.formGroupBox.setLayout(layout)
        self.setupProjectWidgets(layout)
        self.setupTrainingWidgets()
        training_widgets = self.getWidgetsByIndex(0, 4)
        self.addRowsToLayout(layout, training_widgets)

        self.flowGroupBox = QGroupBox("Select Predict Parameters")
        layout2 = QFormLayout()
        self.setupPredictWidgets()
        predict_widgets = self.getWidgetsByIndex(5, 7)
        self.addRowsToLayout(layout2, predict_widgets)
        self.flowGroupBox.setLayout(layout2)
        self.flowGroupBox.hide()

        self.trainGroupBox = QGroupBox("Select Training Parameters")
        layout3 = QFormLayout()
        self.setupHyperparameterWidgets()
        hyperparameter_widgets = self.getWidgetsByIndex(7, 19)
        self.addRowsToLayout(layout3, hyperparameter_widgets)
        self.trainGroupBox.setLayout(layout3)

        self.setupMainLayout()
        self.setWindowTitle("BEAGLES - Machine Learning Tool")