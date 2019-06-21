from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .net.build import TFNet
from .labelFile import LabelFile
from .utils.flags import Flags
import multiprocessing
import os
import re
import time

FLAGS = Flags()


# TODO Build TFNet in a separate process as memory allocated to tf.Session cannot be freed without exit()ing
# TODO Use subprocess.Popen and pass FLAGS

class flowThread(QThread):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, tfnet, flags):
        super(flowThread, self).__init__(parent)
        self.tfnet = tfnet
        self.flags = flags

    def stop(self):
        self.tfnet.FLAGS.kill = True

    def run(self):
        self.tfnet = self.tfnet(self.flags)
        if self.flags.train:
            self.tfnet.train()
        if self.flags.savepb:
            self.tfnet.savepb()
        if not self.flags.fbf == "":
            self.tfnet.annotate()

class flowPrgThread(QThread):

    def __init__(self, parent, flowprg):
        super(flowPrgThread, self).__init__(parent)
        self.flowprg = flowprg

    def run(self):
        while FLAGS.killed is not True and FLAGS.done is not True:
            if round(FLAGS.progress)-1 > self.flowprg.value():
                self.flowprg.setValue(FLAGS.progress)
            time.sleep(0.5)
        if FLAGS.killed is True or FLAGS.done is True:
            self.flowprg.reset()
            return


class flowDialog(QDialog):

    def __init__(self, parent=None):
        super(flowDialog, self).__init__(parent)
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
        layout.addRow(QLabel("Verbose Logging"), self.verbaliseChb)

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

        if self.flowCmb.currentText() == "Flow":  # predict method uses ThreadPool
            tfnet = TFNet(FLAGS)
            tfnet.predict()
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
            tfnet = TFNet(FLAGS)
        if self.flowCmb.currentText() == "Demo":  # OpenCV does not play nice when called outside a main thread
            FLAGS.demo = "camera"
            tfnet = TFNet(FLAGS)
            tfnet.camera()
        self.buttonOk.setEnabled(False)
        if [self.flowCmb.currentText() == "Train" or "Freeze"]:
            self.flowthread = flowThread(self, tfnet=TFNet, flags=FLAGS)
            self.flowthread.setTerminationEnabled(True)
            self.flowthread.finished.connect(self.on_finished)
            self.flowthread.start(priority=5)
            self.flowprgthread = flowPrgThread(self, flowprg=self.flowPrg)
            self.flowprgthread.start(priority=5)

    @pyqtSlot()
    def closeEvent(self, event):
        try:
            self.flowthread.stop()
        except AttributeError:
            pass
        self.buttonOk.setDisabled(False)
        return

    @pyqtSlot()
    def on_finished(self):
        self.buttonOk.setDisabled(False)
        self.findCkpt()
        if FLAGS.train and FLAGS.done:
            form = "Training finished after {} images processed"
            QMessageBox.question(self, 'Success',
                                 form.format((FLAGS.progress / 100) * FLAGS.size * FLAGS.epoch),
                                 QMessageBox.Ok)

    # HELPERS
    def listFiles(self, dir):
        path = QDir(dir)
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
