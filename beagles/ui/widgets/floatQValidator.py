from beagles.base.constants import FLOAT_RE
from PyQt5.QtGui import QValidator


class FloatQValidator(QValidator):
    def __init__(self):
        super(FloatQValidator, self).__init__()
        self.regex = FLOAT_RE

    def validFloatString(self, string):
        match = self.regex.search(string)
        return match.groups()[0] == string if match else False

    def validate(self, string, position):
        if self.validFloatString(string):
            state = self.Acceptable
        elif string == "" or string[position - 1] in 'e.-+':
            state = self.Intermediate
        else:
            state = self.Invalid
        return state, string, position

    def fixup(self, text):
        match = self.regex.search(text)
        return match.groups()[0] if match else ""
