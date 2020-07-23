from unittest import TestCase, mock
import os
import glob
from BEAGLES import get_main_app
import argparse
from libs.utils.flags import Flags
from libs.scripts.voc_to_yolo import convertAnnotation
from libs.project import ProjectDialog
from PyQt5.QtWidgets import QMainWindow


class TestMainWindow(TestCase):

    app = None
    win = None

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    defaultFilename='data/sample_img/sample_dog.jpg',
                    defaultPredefClassFile=Flags().labels,
                    defaultSaveDir=None))
    def setUp(self, args):
        self.app, self.win = get_main_app()

    def testCanvas(self):
        self.canvas = self.win.canvas
        self.assertRaises(AssertionError, self.canvas.resetAllLines)

    def testToggleAdvancedMode(self):
        self.win.toggleAdvancedMode()
        self.win.toggleAdvancedMode()

    def testChangeFormat(self):
        self.win.change_format()
        self.win.change_format()

    def testToggleDrawMode(self):
        self.win.toggleDrawMode()
        self.win.toggleDrawMode()

    def testTrainModel(self):
        self.win.trainModel()

    def testLoadPascalXMLByFilename(self):
        self.win.loadPascalXMLByFilename('test.xml')

    def testFileLoadZoom(self):
        self.win.loadFile('data/sample_img/sample_dog.jpg')
        self.win.setFitWindow()
        self.win.setFitWidth()
        self.win.setZoom(50)
        self.win.closeFile()

    def testImportDirImages(self):
        self.win.importDirImages('data/sample_img')
        self.win.openNextImg()
        self.win.openPrevImg()

    def testImpVideo(self):
        import BEAGLES
        BEAGLES.frame_capture('test.mp4')
        files = glob.glob('*.jpg')
        for file in files:
            os.remove(file)

    def testClearSandbox(self):
        self.win.project.clear_sandbox()

    def tearDown(self):
        self.win.close()
        self.app.quit()

    def test_noop(self):
        pass
