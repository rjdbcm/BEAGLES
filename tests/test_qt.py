from unittest import TestCase, mock
import os
import sys
import glob
from beagles.ui.functions.fileFunctions import FileFunctions
from app import get_main_app
from beagles.io.logs import get_logger
import argparse
from beagles.base.flags import Flags

class TestMainWindow(TestCase):

    app = None
    win = None
    flags = Flags()
    labels = flags.labels

    @classmethod
    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    filename='data/sample_img/sample_dog.jpg',
                    predefined_class_file=labels,
                    save_directory='tests',
                    darkmode=True))
    def setUpClass(cls, args):
        cls.app, cls.win = get_main_app()
        cls.log = get_logger()
        cls.log.info('Setting up Qt app unittests')

    def testCanvas(self):
        self.assertRaises(AssertionError, self.win.canvas.resetAllLines)

    def testToggleAdvancedMode(self):
        self.assertTrue(self.win.beginner())
        self.win.advancedMode()
        self.assertFalse(self.win.beginner())
        self.win.advancedMode(False)
        self.assertTrue(self.win.beginner())

    def testChangeFormat(self):
        self.win.changeFormat()
        self.assertTrue(self.win.usingYoloFormat)
        self.win.changeFormat()
        self.assertTrue(self.win.usingPascalVocFormat)

    def testToggleDrawMode(self):
        self.win.toggleDrawMode(True)
        self.assertTrue(self.win.canvas.editing)

    def testTrainModel(self):
        self.win.trainModel()

    def testLoadPascalXMLByFilename(self):
        self.win.loadPascalXMLByFilename('tests/resources/test.xml')

    def testFileLoadZoom(self):
        self.win.loadFile('data/sample_img/sample_dog.jpg')
        self.win.setFitWin()
        self.win.setFitWidth()
        self.win.setZoom(50)
        self.win.closeFile()

    def testImportDirImages(self):
        self.win.importDirImages('data/sample_img')
        self.win.nextImg()
        self.win.prevImg()

    def testImpVideo(self):
        FileFunctions().frameCapture(os.path.abspath('tests/resources/test.mp4'))
        files = glob.glob('tests/resources/test_frame_*.jpg')
        self.assertFalse(files == [])
        for file in files:
            os.remove(file)

    def tearDown(self) -> None:
        self.win.setClean()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()
        cls.app.quit()

    def test_noop(self):
        pass
