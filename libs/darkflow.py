from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .net.build import TFNet
from .labelFile import LabelFile
import os

DEFAULT_FLAGS = dict(
    train=False,
    savepb=False,
    demo='',
    fbf='',
    trainer='',
    momentum=0.0,
    keep=20,
    batch=16,
    epoch=64,
    save=16000,
    lr=1e-5,
    clip=False,
    saveVideo='out.avi',
    queue=1,
    lb=0.0,
    pbLoad='',
    metaLoad='',
    load=-1,
    model='',
    json=False,
    gpu=0.0,
    gpuName='/gpu:0',
    threshold=0.1,
    verbalise=True
)

DEFAULT_DIRS = dict(
    imgdir='./data/sample_img/',
    binary='./data/bin/',
    config='./data/cfg/',
    dataset='./data/committedframes/',
    backup='./data/ckpt/',
    labels='./data/predefined_classes.txt',
    annotation='./data/committedframes/',
    summary='./data/summaries/'
)


class FlagDict(dict):
    """Allows you to set dict values like attributes"""
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


FLAGS = FlagDict({**DEFAULT_FLAGS, **DEFAULT_DIRS})


class trainThread(QThread):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, tfnet, flags):
        super(trainThread, self).__init__(parent)
        self.tfnet = tfnet
        self.flags = flags

    def run(self):
        self.tfnet = self.tfnet(self.flags)
        self.tfnet.train()

class flowDialog(QDialog):

    def __init__(self, parent=None):
        super(flowDialog, self).__init__(parent)
        self.createFormGroupBox()

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("SLGR-Suite - Machine Learning Tool")

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Select Model and Parameters")
        layout = QFormLayout()

        self.flowCmb = QComboBox()
        self.flowCmb.addItems(["Flow", "Train", "Freeze", "Demo", "Annotate"])
        layout.addRow(QLabel("Select Mode"), self.flowCmb)

        self.modelCmb = QComboBox()
        self.modelCmb.addItems(self.listFiles(FLAGS.config))
        self.modelCmb.setToolTip("Choose a model configuration")
        layout.addRow(QLabel("Model"), self.modelCmb)

        self.loadCmb = QComboBox()
        self.loadCmb.addItem("Last")
        self.loadCmb.addItems(self.listFiles(FLAGS.backup))
        self.loadCmb.setToolTip("Choose a model configuration")
        layout.addRow(QLabel("Checkpoint"), self.loadCmb)

        self.trainerCmb = QComboBox()
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad", "adagradDA", "momentum", "adam", "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        layout.addRow(QLabel("Training Algorithm"), self.trainerCmb)

        self.thresholdSpd = QDoubleSpinBox()
        self.thresholdSpd.setRange(0.0, .99)
        self.thresholdSpd.setSingleStep(0.01)
        self.thresholdSpd.setValue(FLAGS.threshold)
        layout.addRow(QLabel("Confidence Threshold"), self.thresholdSpd)

        self.momentumSpd = QDoubleSpinBox()
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip("Momentum setting for momentum and rmsprop optimizers")
        layout.addRow(QLabel("Momentum"), self.momentumSpd)

        self.keepSpb = QSpinBox()
        self.keepSpb.setValue(FLAGS.keep)
        self.keepSpb.setRange(1, 256)
        layout.addRow(QLabel("Checkpoints to Keep"), self.keepSpb)

        self.batchSpb = QSpinBox()
        self.batchSpb.setRange(2, 256)
        self.batchSpb.setValue(int(FLAGS.batch))
        self.batchSpb.setSingleStep(2)
        layout.addRow(QLabel("Batch Size"), self.batchSpb)

        self.epochSpb = QSpinBox()
        self.epochSpb.setRange(1, 256)
        self.epochSpb.setValue(int(FLAGS.epoch))
        layout.addRow(QLabel("Epochs to Run"), self.epochSpb)

        self.saveSpb = QSpinBox()
        self.saveSpb.setRange(1, 65536)
        self.saveSpb.setValue(FLAGS.save)
        layout.addRow(QLabel("Save Every"), self.saveSpb)

        self.clipChb = QCheckBox()
        layout.addRow(QLabel("Clip Gradients"), self.clipChb)

        self.formGroupBox.setLayout(layout)

    #  SIGNALS

    def trainerSelect(self):
        self.momentumSpd.setDisabled(True)
        for trainer in ("rmsprop", "momentum"):
            if self.trainerCmb.currentText() == trainer:
                self.momentumSpd.setDisabled(False)


    def accept(self):
        #  Set the FLAGS
        FLAGS.model = os.path.join(FLAGS.config, self.modelCmb.currentText())
        FLAGS.load = self.loadCmb.currentText()
        FLAGS.trainer = self.trainerCmb.currentText()
        FLAGS.threshold = self.thresholdSpd.value()
        FLAGS.clip = bool(self.clipChb.checkState())
        FLAGS.momentum = self.momentumSpd.value()
        FLAGS.keep = self.keepSpb.value()
        FLAGS.batch = self.batchSpb.value()
        FLAGS.save = self.saveSpb.value()
        FLAGS.epoch = self.epochSpb.value()
        if FLAGS.load == "Last":
            FLAGS.load = int(-1)

        if self.flowCmb.currentText() == "Flow":
            tfnet = TFNet(FLAGS)
            tfnet.predict()
        elif self.flowCmb.currentText() == "Train":
            FLAGS.train = True
            if not FLAGS.save % FLAGS.batch == 0:
                QMessageBox.question(self, 'Error',
                                     "The value of 'Save' should be divisible by the value of 'Batch'",
                                     QMessageBox.Ok)
                return
            if not os.listdir(FLAGS.dataset):
                QMessageBox.question(self, 'Error',
                                     "No committed frames found",
                                     QMessageBox.Ok)
                return
            else:
                self.thread = trainThread(self, tfnet=TFNet, flags=FLAGS)
                self.thread.start()
        elif self.flowCmb.currentText() == "Freeze":
            FLAGS.savePb = True
            tfnet = TFNet(FLAGS)
            tfnet.savepb()
        elif self.flowCmb.currentText() == "Demo":
            FLAGS.demo = "camera"
            tfnet = TFNet(FLAGS)
            tfnet.camera()
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

    # HELPERS

    def listFiles(self, dir):
        path = QDir(dir)
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
