import unittest
from beagles.ui.functions.mainWindowFunctions import MainWindowFunctions


class TestLib(unittest.TestCase):

    def test_generateColorByGivingUniceText_noError(self):
        res = MainWindowFunctions.generateColorByText(u'\u958B\u555F\u76EE\u9304')
        self.assertTrue(res.green() >= 0)
        self.assertTrue(res.red() >= 0)
        self.assertTrue(res.blue() >= 0)
