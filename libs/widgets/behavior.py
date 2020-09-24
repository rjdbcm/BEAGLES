from PyQt5.QtWidgets import QDialog, QPushButton, QCheckBox, QSpinBox
from libs.widgets.scientificQDoubleSpinBox import ScientificQDoubleSpinBox, FloatQValidator


class BehaviorAnalysisDialog(QDialog):
    def __init__(self, parent):
        super(BehaviorAnalysisDialog, self).__init__(parent)
        self.buttonClassesFile = QPushButton("Open Classes File")
        self.buttonLoad = QPushButton("Load Annotations")
        self.ordinalChb = QCheckBox()
        self.measureIntervalSpb = QSpinBox()
        self.startTimeSpb = QSpinBox()
