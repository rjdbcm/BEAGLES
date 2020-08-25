from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from libs.io.labelFile import LabelFile
from libs.constants import *
from libs.utils.flags import Flags
from libs.widgets.projectDialog import ProjectDialog
from libs.widgets.backend import BackendDialog, BackendThread
#from libs.scripts.genConfig import genConfigYOLOv2
from subprocess import Popen, PIPE
import sys
import os
import re


class FlowDialog(BackendDialog):

    def __init__(self, parent=None, labelfile=None):
        super(FlowDialog, self).__init__(parent)
        self.labelfile = labelfile
        self.flags = Flags()
        self.project = ProjectDialog(self)
        self.project.accepted.connect(self.set_project_name)
        self.setupDialog()
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

        if self.flowCmb.currentText() == "Annotate":
            self.thresholdSpd.setDisabled(False)
    #
    # def updateAnchors(self):
    #     pass
    #     genConfigYOLOv2()

    def set_project_name(self):
        try:
            self.projectLbl.setText(self.project.name)
        except TypeError:
            print(self.project.name)
            pass

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
                                                       'BEAGLES Predict - '
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
                                                   'BEAGLES Annotate - Choose Video file',
                                                   os.getcwd(), filters, options=options)
            self.flags.video = filename[0]
        if [self.flowCmb.currentText() == "Train"]:
            proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
            self.thread = BackendThread(self, proc=proc, flags=self.flags)
            self.thread.setTerminationEnabled(True)
            self.thread.finished.connect(self.onFinished)
            self.thread.connection.progressUpdate.connect(
                self.updateProgress)
            self.thread.start()
        self.flowPrg.setMaximum(0)
        self.buttonRun.setEnabled(False)
        self.buttonRun.hide()
        self.buttonStop.show()
        self.formGroupBox.setEnabled(False)
        self.trainGroupBox.setEnabled(False)

    def stopMessage(self, event):

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
                self.thread.stop()
            except AttributeError:
                pass
            return True

    def closeEvent(self, event):

        def acceptEvent(accepted):
            if accepted:
                self.buttonRun.setDisabled(False)
                self.buttonStop.hide()
                self.buttonRun.show()
                self.flowGroupBox.setEnabled(True)
                self.trainGroupBox.setEnabled(True)
                self.formGroupBox.setEnabled(True)
                # self.findProject()
                try:
                    event.accept()
                except AttributeError:
                    pass

        try:
            thread_running = self.thread.isRunning()
        except AttributeError:
            thread_running = False
        if thread_running:
            accepted = self.stopMessage(event)
            acceptEvent(accepted)
        else:
            self.flowPrg.setMaximum(100)
            self.flowPrg.reset()
            acceptEvent(True)

    def rolloverLogs(self):
        logs = [self.thread.logfile, self.thread.tf_logfile]
        for log in logs:
            if os.stat(log.baseFilename).st_size > 0:
                log.doRollover()

    def onFinished(self):
        self.flags = self.thread.flags
        if self.flags.error:
            QMessageBox.critical(self, "Error Message", self.flags.error,
                                 QMessageBox.Ok)
            self.rolloverLogs()
        if self.flags.verbalise:
            QMessageBox.information(self, "Debug Message", "Process Stopped:\n"
                                    + "\n".join('{}: {}'.format(k, v)
                                                for k, v in
                                                self.flags.items()),
                                    QMessageBox.Ok)
        self.flowGroupBox.setEnabled(True)
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
