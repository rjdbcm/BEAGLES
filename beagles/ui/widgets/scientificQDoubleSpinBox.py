import re
import sys
from PyQt5.QtWidgets import QDoubleSpinBox
from beagles.ui.widgets.floatQValidator import FloatQValidator


class ScientificQDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(float(sys.float_info.min))
        self.setMaximum(float(sys.float_info.max))
        self.validator = FloatQValidator()
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
        groups = self.validator.regex.search(text).groups()
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