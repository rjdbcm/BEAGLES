from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .net.build import TFNet
from .labelFile import LabelFile
import os
import re
import time

class Flags(dict):
    """Allows you to set dict values like attributes"""
    def __init__(self):
        self.get_defaults()

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def get_defaults(self):
        self.train = False
        self.savepb = False
        self.demo = ''
        self.fbf = ''
        self.trainer = ''
        self.momentum = 0.0
        self.keep = 20
        self.batch = 16
        self.epoch = 64
        self.save = 16000
        self.lr = 1e-5
        self.clip = False
        self.saveVideo = './data/sample_img/out.avi'
        self.queue = 1
        self.lb = 0.0
        self.pbLoad = ''
        self.metaLoad = ''
        self.load = -1
        self.model = ''
        self.json = False
        self.gpu = 0.0
        self.gpuName = '/gpu:0'
        self.threshold = 0.1
        self.verbalise = True
        self.kill = False
        self.killed = False
        self.done = False
        self.progress = 0.0
        self.size = 0
        self.imgdir = './data/sample_img/'
        self.binary = './data/bin/'
        self.config = './data/cfg/'
        self.dataset = './data/committedframes/'
        self.backup = './data/ckpt/'
        self.labels = './data/predefined_classes.txt'
        self.annotation = './data/committedframes/'
        self.summary = './data/summaries/'


FLAGS = Flags()

class flowThread(QThread):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, tfnet, flags, flowprg):
        super(flowThread, self).__init__(parent)
        self.tfnet = tfnet
        self.flags = flags
        self.flowprg = flowprg

    def __del__(self):
        self.wait()

    def stop(self):
        self.flowprg.setValue(0)
        self.flowprg.hide()
        self.tfnet.FLAGS.kill = True

    def run(self): # CLASHES With tf.run
        self.tfnet = self.tfnet(self.flags)
        self.flowprg.show()
        if self.flags.train is True:
            self.tfnet.train()
            while self.tfnet.FLAGS.progress < 100:
                self.flowprg.setValue(self.tfnet.FLAGS.progress)
                time.sleep(0.5)
        elif self.flags.savepb is True:
            self.tfnet.savepb()
        else:
            self.tfnet.predict()

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
        self.flowPrg.hide()

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

    #  SIGNALS

    def findCkpt(self):  ### THIS neeeds to happen on change of config and init instead
        self.loadCmb.clear()
        checkpoints = self.listFiles(FLAGS.backup)
        _model = os.path.splitext(self.modelCmb.currentText())
        l = []
        _regex = re.compile("[0-9]+\.")
        for f in checkpoints:
            print("{}\n{}\n".format(f[:len(_model[0])], _model[0]))
            if f[:len(_model[0])] == _model[0]:
                _ckpt = re.search(_regex, f)
                start, end = _ckpt.span()
                self.loadCmb.addItem(str(f[start:end-1]))
                self.buttonOk.setDisabled(False)
                print(f[start:end-1])
            else:
                self.buttonOk.setDisabled(True)

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
        FLAGS.get_defaults()  # Reset FLAGS to values in DefaultFlags.__init__
        FLAGS.model = os.path.join(FLAGS.config, self.modelCmb.currentText())
        FLAGS.load = int(self.loadCmb.currentText())
        FLAGS.trainer = self.trainerCmb.currentText()
        FLAGS.threshold = self.thresholdSpd.value()
        FLAGS.clip = bool(self.clipChb.checkState())
        FLAGS.momentum = self.momentumSpd.value()
        FLAGS.keep = self.keepSpb.value()
        FLAGS.batch = self.batchSpb.value()
        FLAGS.save = self.saveSpb.value()
        FLAGS.epoch = self.epochSpb.value()

        if self.flowCmb.currentText() == "Flow":
            pass
        elif self.flowCmb.currentText() == "Train":

            if not FLAGS.save % FLAGS.batch == 0:
                QMessageBox.question(self, 'Error',
                                     "The value of 'Save Every' should be divisible by the value of 'Batch Size'",
                                     QMessageBox.Ok)
                return
            if not os.listdir(FLAGS.dataset):
                QMessageBox.question(self, 'Error',
                                     "No committed frames found",
                                     QMessageBox.Ok)
                return
            else:
                FLAGS.train = True
        elif self.flowCmb.currentText() == "Freeze":
            FLAGS.savepb = True
        elif self.flowCmb.currentText() == "Annotate":
            formats = ['*.avi', '*.mp4', '*.wmv', '*.mpeg']
            filters = "Video Files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename = QFileDialog.getOpenFileName(self, 'SLGR-Suite Annotate - Choose Video file', os.getcwd(),
                                                   filters, options=options)
            FLAGS.fbf = filename[0]
            tfnet = TFNet(FLAGS)
            tfnet.annotate()
        elif self.flowCmb.currentText() == "Demo":
            FLAGS.demo = "camera"
            tfnet = TFNet(FLAGS)
            print("zeep")
            tfnet.camera()

        if [self.flowCmb.currentText() == "Flow" or "Train" or "Freeze"]:
            self.buttonOk.setDisabled(True)
            self.buttonOk.update()
            self.flowthread = flowThread(self, tfnet=TFNet, flags=FLAGS, flowprg=self.flowPrg)
            self.flowthread.setTerminationEnabled(True)
            self.flowthread.finished.connect(self.on_finished)
            self.flowthread.start()

    def closeEvent(self, event):
        try:
            self.flowthread.stop()
        except AttributeError:
            print(help(self))
            pass
        self.buttonOk.setDisabled(False)
        return

    @pyqtSlot()
    def on_finished(self):
        self.buttonOk.setDisabled(False)
        self.buttonCancel.setDisabled(False)
        self.flowPrg.setValue(0)
        self.flowPrg.hide()

    @pyqtSlot()
    def on_error(self):
        self.buttonOk.setDisabled(False)
        QMessageBox.question(self, 'Error',
                             "An Error Occurred:\n{}".format(FLAGS),
                             QMessageBox.Ok)

    # HELPERS
    def listFiles(self, dir):
        path = QDir(dir)
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
