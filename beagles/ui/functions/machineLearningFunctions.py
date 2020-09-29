from PyQt5.QtCore import QProcess
from beagles.ui.functions.mainWindowFunctions import MainWindowFunctions
from beagles.ui.widgets.flowDialog import FlowDialog
from beagles.base.flags import Flags


class MachineLearningFunctions(MainWindowFunctions):
    def __init__(self):
        super(MachineLearningFunctions, self).__init__()
        self.tb_process = QProcess(self)
        self.tb_process.start("tensorboard",
                              ["--logdir=data/summaries", "--debugger_port=6064"])
        self.trainDialog = FlowDialog(parent=self, labelfile=Flags().labels)
