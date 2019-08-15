from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .labelFile import LabelFile
from .utils.flags import Flags, FlagIO
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


class MultiCamThread(QThread):
    def __init__(self, parent, model):
        super(MultiCamThread, self).__init__(parent)
        self.devs = []
        self.fps = []
        self.model = model
        self.stopped = False

    def silence(self):
        pass

    def enumDevs(self):
        index = 0
        cv2.redirectError(self.silence)
        while index < 32:
            start = time.time()
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 144)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
            end = time.time()
            elapsed = end - start
            if cap is None or not cap.isOpened():
                pass
            else:
                self.devs.append(index)
                self.fps.append(index / elapsed)
            index += 1
        self.devs = dict(enumerate(self.devs, start=1))
        self.fps = dict(enumerate(self.fps, start=1))
        self.model.clear()
        return self.devs

    def run(self):
        self.model.clear()  # Model is QObject so no issue outside main thread
        self.model.appendRow(QStandardItem("Refreshing..."))
        self.enumDevs()
        self.model.clear()
        for (camera_no, dev_num), (camera_no, fps) in zip(
                self.devs.items(), self.fps.items()):
            item = QStandardItem(
                " ".join(["Camera", str(camera_no), "on",
                          "/dev/video{}".format(dev_num),
                          ]))
            item.setData([dev_num, fps])
            item.setCheckable(True)
            self.model.appendRow(item)
        if not self.model.rowCount():
            self.model.appendRow(QStandardItem("No Devices Found"))


class FlowDialog(QDialog):

    def __init__(self, parent=None, labelfile=None):
        super(FlowDialog, self).__init__(parent)
        self.flags = Flags()
        self.oldBatchValue = int(self.flags.batch)
        self.oldSaveValue = int(self.flags.save)
        # allow use of labels file passed by slgrSuite
        self.labelfile = labelfile
        self.formGroupBox = QGroupBox("Select Model and Checkpoint")
        layout = QFormLayout()

        self.flowCmb = QComboBox()
        self.flowCmb.addItems(
            ["Train", "Predict", "Freeze", "Capture", "Annotate"])
        self.flowCmb.currentIndexChanged.connect(self.flowSelect)
        layout.addRow(QLabel("Mode"), self.flowCmb)

        self.modelCmb = QComboBox()
        self.modelCmb.addItems(self.listFiles(self.flags.config))
        self.modelCmb.setToolTip("Choose a model configuration")
        self.modelCmb.currentIndexChanged.connect(self.findCkpt)
        layout.addRow(QLabel("Model"), self.modelCmb)

        self.loadCmb = QComboBox()
        self.loadCmb.setToolTip("Choose a checkpoint")
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
        self.jsonChb.setChecked(self.flags.json)

        layout2.addRow(QLabel("Output JSON Annotations"), self.jsonChb)

        self.flowGroupBox.setLayout(layout2)
        self.flowGroupBox.hide()

        self.trainGroupBox = QGroupBox("Select Training Parameters")

        layout3 = QFormLayout()

        self.trainerCmb = QComboBox()
        self.trainerCmb.addItems(["rmsprop", "adadelta", "adagrad",
                                  "adagradDA", "momentum", "nesterov", "adam",
                                  "ftrl", "sgd"])
        self.trainerCmb.currentIndexChanged.connect(self.trainerSelect)
        layout3.addRow(QLabel("Training Algorithm"), self.trainerCmb)

        self.momentumSpd = QDoubleSpinBox()
        self.momentumSpd.setRange(0.0, .99)
        self.momentumSpd.setSingleStep(0.01)
        self.momentumSpd.setToolTip("Momentum setting for momentum and "
                                    "rmsprop optimizers")
        layout3.addRow(QLabel("Momentum"), self.momentumSpd)

        self.learningRateSpd = ScientificDoubleSpinBox()
        self.learningRateSpd.setValue(self.flags.lr)
        layout3.addRow(QLabel("Initial Learning Rate"), self.learningRateSpd)

        self.maxLearningRateSpd = ScientificDoubleSpinBox()
        self.maxLearningRateSpd.setValue(self.flags.max_lr)
        layout3.addRow(QLabel("Maximum Learning Rate"), self.maxLearningRateSpd)

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

        self.clipChb = QCheckBox()
        layout3.addRow(QLabel("Clip Gradients"), self.clipChb)

        self.trainGroupBox.setLayout(layout3)

        self.demoGroupBox = QGroupBox("Select Capture Parameters")
        layout4 = QFormLayout()

        self.timeoutTme = QTimeEdit()
        self.timeoutTme.setDisplayFormat('hh:mm:ss')
        layout4.addRow(QLabel('Record Time (hh:mm:ss)'), self.timeoutTme)

        self.grayscaleChb = QCheckBox()
        self.grayscaleChb.setChecked(self.flags.grayscale)
        layout4.addRow(QLabel('Convert to Grayscale'), self.grayscaleChb)

        self.lineFrm = QFrame()
        self.lineFrm.setFrameShape(QFrame.HLine)
        self.lineFrm.setFrameShadow(QFrame.Sunken)
        layout4.addRow(self.lineFrm)

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

        self.flowPrg = QProgressBar()
        self.flowPrg.setRange(0, 100)
        self.buttonOk = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonCancel = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonStop = QPushButton("Stop")
        self.buttonStop.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        self.buttonStop.hide()
        self.buttonOk.accepted.connect(self.accept)
        self.buttonStop.clicked.connect(self.closeEvent)
        self.buttonCancel.rejected.connect(self.close)

        main_layout = QGridLayout()
        main_layout.addWidget(self.formGroupBox, 0, 0)
        main_layout.addWidget(self.flowGroupBox, 1, 0)
        main_layout.addWidget(self.demoGroupBox, 2, 0)
        main_layout.addWidget(self.trainGroupBox, 3, 0)
        main_layout.setSizeConstraint(QLayout.SetFixedSize)
        main_layout.addWidget(self.buttonOk, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonStop, 4, 0, Qt.AlignRight)
        main_layout.addWidget(self.buttonCancel, 4, 0, Qt.AlignLeft)
        main_layout.addWidget(self.flowPrg, 4, 0, Qt.AlignCenter)
        self.setLayout(main_layout)

        self.setWindowTitle("SLGR-Suite - Machine Learning Tool")
        self.findCkpt()

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
                self.buttonOk.setDisabled(False)
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
        fh = open(file, 'r')
        data = fh.read()
        fh.close()
        result = regex.sub('"{}"'.format(replacement), data)
        fh = open(file, 'w')
        fh.write(result)
        fh.close()

    def listCameras(self):
        self.refreshDevBtn.setDisabled(True)
        self.buttonOk.setDisabled(True)
        model = self.deviceItemModel
        t = MultiCamThread(self, model)
        if t.isRunning():
            return
        else:
            self.flowPrg.setMaximum(0)
            t.start()
            t.finished.connect(self._list_cameras_finished)

    def _list_cameras_finished(self):
        self.refreshDevBtn.setDisabled(False)
        self.buttonOk.setDisabled(False)
        self.flowPrg.setMaximum(100)

    def trainerSelect(self):
        self.momentumSpd.setDisabled(True)
        for trainer in ("rmsprop", "momentum", "nesterov"):
            if self.trainerCmb.currentText() == trainer:
                self.momentumSpd.setDisabled(False)

    def flowSelect(self):
        if self.flowCmb.currentText() == "Capture":
            self.demoGroupBox.show()
        else:
            self.demoGroupBox.hide()

        if self.flowCmb.currentText() == "Predict":
            self.flowGroupBox.show()
        else:
            self.flowGroupBox.hide()

        if self.flowCmb.currentText() == "Freeze":
            self.thresholdSpd.setDisabled(True)
        else:
            self.thresholdSpd.setDisabled(False)

        if self.flowCmb.currentText() == "Train":
            self.trainGroupBox.show()
            self.thresholdSpd.setDisabled(True)
        else:
            self.trainGroupBox.hide()
            self.loadCmb.setCurrentIndex(0)

    def accept(self):
        """set flags for darkflow and prevent startup if errors anticipated"""
        self.updateCkptFile()  # Make sure TFNet gets the correct checkpoint
        self.flags.get_defaults()  # Reset self.flags
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
        self.flags.verbalise = bool(self.verbaliseChb.checkState())
        self.flags.momentum = self.momentumSpd.value()
        self.flags.lr = self.learningRateSpd.value()
        self.flags.keep = self.keepSpb.value()
        self.flags.batch = self.batchSpb.value()
        self.flags.save = self.saveSpb.value()
        self.flags.epoch = self.epochSpb.value()
        self.flags.labels = self.labelfile  # use labelfile set by slgrSuite
        self.flags.json = bool(self.jsonChb.checkState()) if \
            self.flowGroupBox.isEnabled() else self.flags.json
        self.flags.timeout = QTime(0, 0, 0).secsTo(self.timeoutTme.time())
        fps_list = list()
        for i in range(self.deviceItemModel.rowCount()):
            item = self.deviceItemModel.item(i)
            if item.checkState():
                self.flags.capdevs.append(item.data()[0])
                fps_list.append(item.data()[1])
        try:
            self.flags.fps = min(fps_list)
        except ValueError:
            pass

        if not self.flowCmb.currentText() == "Train" and self.flags.load == 0:
            QMessageBox.warning(self, 'Error', "Invalid checkpoint",
                                QMessageBox.Ok)
            return
        if self.flowCmb.currentText() == "Predict":
            self.flowGroupBox.setDisabled(True)
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
        if self.flowCmb.currentText() == "Freeze":
            self.flags.freeze = True
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
            self.flags.fbf = filename[0]
        if self.flowCmb.currentText() == "Capture":
            if not self.flags.capdevs:
                QMessageBox.warning(self, 'Error',
                                    'No capture device is selected',
                                    QMessageBox.Ok)
                return
            if not self.flags.timeout:
                QMessageBox.warning(self, 'Error',
                                    "Please specify a record time",
                                    QMessageBox.Ok)
                return
            self.demoGroupBox.setDisabled(True)
            self.flags.demo = "camera"
        if [self.flowCmb.currentText() == "Train" or "Freeze"]:
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
        self.buttonOk.setEnabled(False)
        self.buttonOk.hide()
        self.buttonStop.show()
        self.formGroupBox.setEnabled(False)
        self.trainGroupBox.setEnabled(False)

    def closeEvent(self, event):

        def acceptEvent():
            self.buttonOk.setDisabled(False)
            self.buttonStop.hide()
            self.buttonOk.show()
            self.flowGroupBox.setEnabled(True)
            self.demoGroupBox.setEnabled(True)
            self.trainGroupBox.setEnabled(True)
            self.formGroupBox.setEnabled(True)
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
        self.buttonOk.setDisabled(False)
        self.buttonStop.hide()
        self.buttonOk.show()
        self.findCkpt()

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
        filters = ["*.cfg", "*.meta"]
        path.setNameFilters(filters)
        files = path.entryList()
        return files
