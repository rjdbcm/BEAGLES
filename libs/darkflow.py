from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .labelFile import LabelFile
from .utils.flags import Flags, FlagIO
import subprocess
import cv2
import sys
import os
import re
import time

FLAGS = Flags()


class FlowThread(QThread, FlagIO):
    """Needed so the long-running train ops don't block Qt UI"""

    def __init__(self, parent, proc, flags, pbar, rate=1):
        super(FlowThread, self).__init__(parent)
        self.pbar = pbar
        self.rate = rate
        self.proc = proc
        self.flags = flags
        self.send_flags()
        time.sleep(1)

    def returnFlags(self):
        global FLAGS
        FLAGS = self.read_flags()

    def stop(self):
        if not self.flags.done:
            self.flags.kill = True
            self.io_flags()
        self.read_flags()
        self.logger.info('Thread killed')
        self.pbar.reset()
        self.proc.kill()
        self.returnFlags()
        self.logfile.doRollover()
        self.cleanup_ramdisk()

    def run(self):
        first = True
        incr = False
        while self.proc.poll() is None:
            # pulse flowPrg during startup
            if first:
                self.pbar.setRange(0, 0)
                first = False
            # stop pulsing flowPrg once the progress has increased
            if incr:
                self.pbar.setRange(0, 100)
                incr = False
            if round(self.flags.progress - 1) > self.pbar.value():
                self.pbar.setValue(self.flags.progress)
                incr = True
            time.sleep(self.rate)
            self.read_flags()
            if self.flags.done:
                self.read_flags()
                self.proc.kill()
                self.returnFlags()
                self.cleanup_ramdisk()
                self.logfile.doRollover()
                self.pbar.reset()


class MultiCamThread(QThread):
    def __init__(self, parent, model, pbar):
        super(MultiCamThread, self).__init__(parent)
        self.devs = []
        self.model = model
        self.pbar = pbar
        self.stopped = False

    def enumDevs(self):
        index = 0
        while index < 32:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 144)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
            if not cap.read()[0]:
                pass
            else:
                self.devs.append(index)
            index += 1
        self.devs = dict(enumerate(self.devs))
        self.model.clear()
        self.pbar.setRange(0, 100)
        return self.devs  # Use whilenot-else to display cams in UI

    def run(self):
        self.model.clear()
        self.model.appendRow(QStandardItem("Refreshing..."))
        self.pbar.setRange(0, 0)
        self.enumDevs()
        while not self.devs:
            time.sleep(1)
        else:
            self.model.clear()
            for k, v in self.devs.items():
                item = QStandardItem(" ".join(["Camera",
                                               str(k), "on",
                                               "/dev/video{}".format(v)]))
                item.setData(v)
                item.setCheckable(True)
                self.model.appendRow(item)


class FlowDialog(QDialog):

    def __init__(self, parent=None, labelfile=None):
        super(FlowDialog, self).__init__(parent)

        self.formGroupBox = QGroupBox("Select Model and Checkpoint")
        layout = QFormLayout()

        self.flowCmb = QComboBox()
        self.flowCmb.addItems(
            ["Train", "Flow", "Freeze", "Capture", "Annotate"])
        self.flowCmb.currentIndexChanged.connect(self.flowSelect)
        layout.addRow(QLabel("Mode"), self.flowCmb)

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
        layout.addRow(QLabel("Threshold"), self.thresholdSpd)

        self.verbaliseChb = QCheckBox()
        layout.addRow(QLabel("Verbose"), self.verbaliseChb)

        self.formGroupBox.setLayout(layout)

        self.flowGroupBox = QGroupBox("Select Flow Parameters")

        layout2 = QFormLayout()

        self.jsonChb = QCheckBox()

        layout2.addRow(QLabel("Output JSON Annotations"), self.jsonChb)

        self.flowGroupBox.setLayout(layout2)
        self.flowGroupBox.hide()

        self.trainGroupBox = QGroupBox("Select Training Parameters")

        layout3 = QFormLayout()

        self.trainerCmb = QComboBox()
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad",
                                  "adagradDA", "momentum", "adam",
                                  "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        layout3.addRow(QLabel("Training Algorithm"), self.trainerCmb)

        self.momentumSpd = QDoubleSpinBox()
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip("Momentum setting for momentum and "
                                    "rmsprop optimizers")
        layout3.addRow(QLabel("Momentum"), self.momentumSpd)

        self.keepSpb = QSpinBox()
        self.keepSpb.setValue(FLAGS.keep)
        self.keepSpb.setRange(1, 256)
        layout3.addRow(QLabel("Checkpoints to Keep"), self.keepSpb)

        self.batchSpb = QSpinBox()
        self.batchSpb.setRange(2, 256)
        self.batchSpb.setValue(int(FLAGS.batch))
        self.batchSpb.setSingleStep(2)
        layout3.addRow(QLabel("Batch Size"), self.batchSpb)

        self.epochSpb = QSpinBox()
        self.epochSpb.setRange(1, 65536)
        self.epochSpb.setValue(int(FLAGS.epoch))
        layout3.addRow(QLabel("Epochs to Run"), self.epochSpb)

        self.saveSpb = QSpinBox()
        self.saveSpb.setRange(1, 65536)
        self.saveSpb.setValue(FLAGS.save)
        layout3.addRow(QLabel("Save Every"), self.saveSpb)

        self.clipChb = QCheckBox()
        layout3.addRow(QLabel("Clip Gradients"), self.clipChb)

        self.trainGroupBox.setLayout(layout3)

        self.demoGroupBox = QGroupBox("Select Capture Parameters")
        layout4 = QFormLayout()

        self.deviceLbl = QLabel("Available Video Devices:")
        layout4.addRow(self.deviceLbl)

        self.deviceLsV = QListView()
        self.deviceItemModel = QStandardItemModel()
        self.deviceLsV.setModel(self.deviceItemModel)
        layout4.addRow(self.deviceLsV)

        self.refreshDevBtn = QPushButton()
        self.refreshDevBtn.setText("Refresh Device List")
        self.refreshDevBtn.clicked.connect(self.listCameras)
        layout4.addRow(self.refreshDevBtn)

        self.demoGroupBox.setLayout(layout4)
        self.demoGroupBox.hide()

        self.labelfile = labelfile

        self.buttonOk = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonCancel = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonOk.accepted.connect(self.accept)
        self.buttonCancel.rejected.connect(self.close)
        self.flowPrg = QProgressBar()
        self.flowPrg.setRange(0, 100)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.flowGroupBox, 1, 0)
        main_layout.addWidget(self.demoGroupBox, 2, 0)
        main_layout.addWidget(self.trainGroupBox, 3, 0)
        main_layout.setSizeConstraint(QLayout.SetFixedSize)
        main_layout.addWidget(self.buttonOk, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.flowPrg, 4, 0, Qt.AlignCenter)
        self.setLayout(main_layout)

        self.setWindowTitle("SLGR-Suite - Machine Learning Tool")
        self.findCkpt()

    def findCkpt(self):
        self.loadCmb.clear()
        checkpoints = self.listFiles(FLAGS.backup)
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
                self.buttonOk.setDisabled(False)
            # else:
            #     self.buttonOk.setDisabled(True)
        l = list(map(int, l))
        l.sort(reverse=True)
        l = list(map(str, l))
        self.loadCmb.addItems(l)

    def listCameras(self):
        self.refreshDevBtn.setDisabled(True)
        self.buttonOk.setDisabled(True)
        model = self.deviceItemModel
        pbar = self.flowPrg
        t = MultiCamThread(self, model, pbar)
        if t.isRunning():
            return
        else:
            t.start()
            t.finished.connect(self._list_cameras_finished)

    def _list_cameras_finished(self):
        self.refreshDevBtn.setDisabled(False)
        self.buttonOk.setDisabled(False)

    def trainerSelect(self):
        self.momentumSpd.setDisabled(True)
        for trainer in ("rmsprop", "momentum"):
            if self.trainerCmb.currentText() == trainer:
                self.momentumSpd.setDisabled(False)

    def flowSelect(self):
        if self.flowCmb.currentText() == "Capture":
            self.demoGroupBox.show()
        else:
            self.demoGroupBox.hide()

        if self.flowCmb.currentText() == "Flow":
            self.flowGroupBox.show()
        else:
            self.flowGroupBox.hide()

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
        """set flags for darkflow and prevent startup if errors anticipated"""
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
        FLAGS.json = bool(self.jsonChb.checkState()) if \
            self.flowGroupBox.isEnabled() else FLAGS.json

        for i in range(self.deviceItemModel.rowCount()):
            item = self.deviceItemModel.item(i)
            if item.checkState():
                FLAGS.capdevs.append(item.data())

        if not self.flowCmb.currentText() == "Train" and FLAGS.load == 0:
            QMessageBox.warning(self, 'Error', "Invalid checkpoint",
                                 QMessageBox.Ok)
            return
        if self.flowCmb.currentText() == "Flow":
            pass
        if self.flowCmb.currentText() == "Train":
            if not FLAGS.save % FLAGS.batch == 0:
                QMessageBox.warning(self, 'Error',
                                     "The value of 'Save Every' should be "
                                     "divisible by the value of 'Batch Size'",
                                     QMessageBox.Ok)
                return
            dataset = [f for f in os.listdir(FLAGS.dataset)
                       if not f.startswith('.')]
            if not dataset:
                QMessageBox.warning(self, 'Error',
                                     'No frames or annotations found',
                                     QMessageBox.Ok)
                return
            else:
                FLAGS.train = True
        if self.flowCmb.currentText() == "Freeze":
            FLAGS.savepb = True
        if self.flowCmb.currentText() == "Annotate":
            formats = ['*.avi', '*.mp4', '*.wmv', '*.mpeg']
            filters = "Video Files (%s)" % ' '.join(
                formats + ['*%s' % LabelFile.suffix])
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename = QFileDialog.getOpenFileName(self,
                                                   'SLGR-Suite Annotate - '
                                                   'Choose Video file',
                                                   os.getcwd(),
                                                   filters, options=options)
            FLAGS.fbf = filename[0]
        if self.flowCmb.currentText() == "Capture":
            if not FLAGS.capdevs:
                QMessageBox.warning(self, 'Error',
                                     'No capture device is selected',
                                     QMessageBox.Ok)
                return
            FLAGS.demo = "camera"
        if [self.flowCmb.currentText() == "Train" or "Freeze"]:
            proc = subprocess.Popen([sys.executable, os.path.join(
                os.getcwd(), "libs/scripts/wrapper.py")],
                                    stdout=subprocess.PIPE, shell=False)
            self.flowthread = FlowThread(self, proc=proc, flags=FLAGS,
                                         pbar=self.flowPrg)
            self.flowthread.setTerminationEnabled(True)
            self.flowthread.finished.connect(self.onFinished)
            self.flowthread.start()
        self.buttonOk.setEnabled(False)
        self.formGroupBox.setEnabled(False)
        self.trainGroupBox.setEnabled(False)

    def closeEvent(self, event):
        try:
            thread_running = self.flowthread.isRunning()
        except AttributeError:
            thread_running = False
        if thread_running:
            msg = "Are you sure you want to close this dialog? " \
                  "This will kill any running processes."
            reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Yes,
                                         QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
            else:
                try:
                    self.flowthread.stop()
                except AttributeError:
                    pass
                self.buttonOk.setDisabled(False)
                self.trainGroupBox.setEnabled(True)
                self.formGroupBox.setEnabled(True)
                event.accept()
        else:
            self.buttonOk.setDisabled(False)
            self.trainGroupBox.setEnabled(True)
            self.formGroupBox.setEnabled(True)
            event.accept()

    def onFinished(self):
        if FLAGS.error:
            QMessageBox.critical(self, "Error Message", FLAGS.error,
                              QMessageBox.Ok)
        if FLAGS.verbalise:
            QMessageBox.information(self, "Debug Message", "Process Stopped:\n"
                                    + "\n".join('{}: {}'.format(k, v)
                                                for k, v in FLAGS.items()),
                                    QMessageBox.Ok)
        self.trainGroupBox.setEnabled(True)
        self.formGroupBox.setEnabled(True)
        self.buttonOk.setDisabled(False)
        self.findCkpt()

    # HELPERS
    @staticmethod
    def listFiles(path):
        path = QDir(path)
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
