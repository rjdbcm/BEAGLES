from unittest import TestCase, mock
from slgrSuite import get_main_app
import argparse
from libs.utils.flags import Flags


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

    def tearDown(self):
        self.win.close()
        self.app.quit()

    def test_noop(self):
        pass
