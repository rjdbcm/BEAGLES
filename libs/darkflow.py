from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .labelFile import LabelFile
from .utils.flags import Flags, FlagIO
import subprocess
import sys
import os
import re
import time

FLAGS = Flags()


class flowThread(QThread, FlagIO):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, proc, flags, pbar, rate=1):
        super(flowThread, self).__init__(parent)
        self.pbar = pbar
        self.rate = rate
        self.proc = proc
        self.flags = flags
        self.send_flags()
        time.sleep(1)

    def stop(self):
        if not self.flags.done:
            self.flags.kill = True
            self.io_flags()
        self.read_flags()
        self.pbar.reset()
        self.proc.kill()
        self.cleanup_ramdisk()

    def run(self):
        while self.proc.poll() is None:
            if round(self.flags.progress-1) > self.pbar.value():
                self.pbar.setValue(self.flags.progress)
            time.sleep(self.rate)
            self.read_flags()
            if self.flags.done:
                self.read_flags()
                self.cleanup_ramdisk()
                self.pbar.reset()


class flowDialog(QDialog):

    def __init__(self, parent=None, labelfile=None):
        super(flowDialog, self).__init__(parent)
        self.labelfile = labelfile
        self.createFormGroupBox()

        self.buttonOk = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonCancel = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonOk.accepted.connect(self.accept)
        self.buttonCancel.rejected.connect(self.close)
        self.flowPrg = QProgressBar()
        self.flowPrg.setRange(0, 100)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.trainGroupBox)
        mainLayout.setSizeConstraint(QLayout.SetFixedSize)
        mainLayout.addWidget(self.buttonOk, 2, 0, Qt.AlignRight)
        mainLayout.addWidget(self.buttonCancel, 2, 0, Qt.AlignLeft)
        mainLayout.addWidget(self.flowPrg, 2, 0, Qt.AlignCenter)
        self.setLayout(mainLayout)

        self.setWindowTitle("SLGR-Suite - Machine Learning Tool")
        self.findCkpt()

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Select Model and Checkpoint")
        layout = QFormLayout()

        self.flowCmb = QComboBox()
        self.flowCmb.addItems(["Train", "Flow", "Freeze", "Demo", "Annotate"])
        self.flowCmb.currentIndexChanged.connect(self.flowSelect)
        layout.addRow(QLabel("Select Mode"), self.flowCmb)

        self.modelCmb = QComboBox()
        self.modelCmb.addItems(self.listFiles(FLAGS.config))
        self.modelCmb.setToolTip("Choose a model configuration")
        self.modelCmb.currentIndexChanged.connect(self.findCkpt)
        layout.addRow(QLabel("Model"), self.modelCmb)

        self.loadCmb = QComboBox()
        self.loadCmb.setToolTip("Choose a checkpoint")
        layout.addRow(QLabel("Checkpoint"), self.loadCmb)

        self.thresholdSpd = QDoubleSpinBox()
        self.thresholdSpd.setRange(0.0, .99)
        self.thresholdSpd.setSingleStep(0.01)
        self.thresholdSpd.setValue(FLAGS.threshold)
        layout.addRow(QLabel("Confidence Threshold"), self.thresholdSpd)

        self.verbaliseChb = QCheckBox()
        layout.addRow(QLabel("Verbose Messages & Log"), self.verbaliseChb)

        self.formGroupBox.setLayout(layout)

        self.trainGroupBox = QGroupBox("Select Training Parameters")

        layout2 = QFormLayout()

        self.trainerCmb = QComboBox()
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad", "adagradDA", "momentum", "adam", "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        layout2.addRow(QLabel("Training Algorithm"), self.trainerCmb)

        self.momentumSpd = QDoubleSpinBox()
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip("Momentum setting for momentum and rmsprop optimizers")
        layout2.addRow(QLabel("Momentum"), self.momentumSpd)

        self.keepSpb = QSpinBox()
        self.keepSpb.setValue(FLAGS.keep)
        self.keepSpb.setRange(1, 256)
        layout2.addRow(QLabel("Checkpoints to Keep"), self.keepSpb)

        self.batchSpb = QSpinBox()
        self.batchSpb.setRange(2, 256)
        self.batchSpb.setValue(int(FLAGS.batch))
        self.batchSpb.setSingleStep(2)
        layout2.addRow(QLabel("Batch Size"), self.batchSpb)

        self.epochSpb = QSpinBox()
        self.epochSpb.setRange(1, 256)
        self.epochSpb.setValue(int(FLAGS.epoch))
        layout2.addRow(QLabel("Epochs to Run"), self.epochSpb)

        self.saveSpb = QSpinBox()
        self.saveSpb.setRange(1, 65536)
        self.saveSpb.setValue(FLAGS.save)
        layout2.addRow(QLabel("Save Every"), self.saveSpb)

        self.clipChb = QCheckBox()
        layout2.addRow(QLabel("Clip Gradients"), self.clipChb)

        self.trainGroupBox.setLayout(layout2)

    def findCkpt(self):
        self.loadCmb.clear()
        checkpoints = self.listFiles(FLAGS.backup)
        _model = os.path.splitext(self.modelCmb.currentText())
        l = ['0']
        _regex = re.compile("\-[0-9]+\.")  # a dash followed by a number or numbers followed by a dot
        for f in checkpoints:
            if f[:len(_model[0])] == _model[0]:
                _ckpt = re.search(_regex, f)
                start, end = _ckpt.span()
                n = f[start+1:end-1]
                l.append(n)
                self.buttonOk.setDisabled(False)
            # else:
            #     self.buttonOk.setDisabled(True)
        l = list(map(int, l))
        l.sort(reverse=True)
        l = list(map(str, l))
        self.loadCmb.addItems(l)

    def trainerSelect(self):
        self.momentumSpd.setDisabled(True)
        for trainer in ("rmsprop", "momentum"):
            if self.trainerCmb.currentText() == trainer:
                self.momentumSpd.setDisabled(False)

    def flowSelect(self):
        if not self.flowCmb.currentText() == "Train":
            self.trainGroupBox.hide()
            self.loadCmb.setCurrentIndex(0)
            if self.flowCmb.currentText() == "Freeze":
                self.thresholdSpd.setDisabled(True)
            else:
                self.thresholdSpd.setDisabled(False)
        else:
            self.trainGroupBox.show()
            self.thresholdSpd.setDisabled(False)

    def accept(self):
        FLAGS.get_defaults()  # Reset FLAGS
        FLAGS.model = os.path.join(FLAGS.config, self.modelCmb.currentText())
        try:
            FLAGS.load = int(self.loadCmb.currentText())
        except ValueError:
            FLAGS.load = 0
            pass
        FLAGS.trainer = self.trainerCmb.currentText()
        FLAGS.threshold = self.thresholdSpd.value()
        FLAGS.clip = bool(self.clipChb.checkState())
        FLAGS.verbalise = bool(self.verbaliseChb.checkState())
        FLAGS.momentum = self.momentumSpd.value()
        FLAGS.keep = self.keepSpb.value()
        FLAGS.batch = self.batchSpb.value()
        FLAGS.save = self.saveSpb.value()
        FLAGS.epoch = self.epochSpb.value()
        FLAGS.labels = self.labelfile

        if self.flowCmb.currentText() == "Flow":
            pass
        if self.flowCmb.currentText() == "Train":
            if not FLAGS.save % FLAGS.batch == 0:
                QMessageBox.question(self, 'Error',
                                     "The value of 'Save Every' should be divisible by the value of 'Batch Size'",
                                     QMessageBox.Ok)
                return
            if not os.listdir(FLAGS.dataset):
                QMessageBox.question(self, 'Error', "No committed frames found", QMessageBox.Ok)
                return
            else:
                FLAGS.train = True
        if self.flowCmb.currentText() == "Freeze":
            FLAGS.savepb = True
        if self.flowCmb.currentText() == "Annotate":  # OpenCV does not play nice when called outside a main thread
            formats = ['*.avi', '*.mp4', '*.wmv', '*.mpeg']
            filters = "Video Files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename = QFileDialog.getOpenFileName(self, 'SLGR-Suite Annotate - Choose Video file', os.getcwd(),
                                                   filters, options=options)
            FLAGS.fbf = filename[0]
        if self.flowCmb.currentText() == "Demo":  # OpenCV does not play nice when called outside a main thread
            FLAGS.demo = "camera"
        self.buttonOk.setEnabled(False)
        if [self.flowCmb.currentText() == "Train" or "Freeze"]:
            proc = subprocess.Popen([sys.executable, os.path.join(os.getcwd(), "libs/wrapper/wrapper.py")], stdout=subprocess.PIPE, shell=False)
            self.flowthread = flowThread(self, proc=proc, flags=FLAGS, pbar=self.flowPrg)
            self.flowthread.setTerminationEnabled(True)
            self.flowthread.finished.connect(self.on_finished)
            self.flowthread.start()


    @pyqtSlot()
    def closeEvent(self, event):

        if self.flowthread.isRunning():
            msg = "Are you sure you want to close this dialog? This will kill any running processes."
            reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
            else:
                try:
                    self.flowthread.stop()
                except AttributeError:
                    pass
                self.buttonOk.setDisabled(False)
                event.accept()
        else:
            self.buttonOk.setDisabled(False)
            event.accept()


    @pyqtSlot()
    def on_finished(self):
        if FLAGS.verbalise:
            QMessageBox.question(self, "Success", "Process Stopped:\n" + "\n".join('{}: {}'.format(k, v) for k, v in FLAGS.items()),
                                 QMessageBox.Ok)
        self.buttonOk.setDisabled(False)
        self.findCkpt()

    # HELPERS
    def listFiles(self, dir):
        path = QDir(dir)
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
