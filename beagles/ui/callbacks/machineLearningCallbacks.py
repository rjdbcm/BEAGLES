import os
import errno
import shutil
import subprocess
import webbrowser
from PyQt5.QtWidgets import QMessageBox
from beagles.ui.functions.machineLearningFunctions import MachineLearningFunctions


class MachineLearningCallbacks(MachineLearningFunctions):

    def visualize(self):
        webbrowser.open_new_tab('http://localhost:6006/#scalars&_smoothingWeight=0')

    def trainModel(self):
        if not self.mayContinue():
            return
        self.trainDialog.show()

    def commitAnnotatedFrames(self):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure you want to commit all "
                                     "open files?", QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.No or not self.mayContinue():
            return
        else:
            pass

        path = self.setDefaultOpenDirPath()
        if path == self.committedframesDataPath:
            self.errorMessage("", "These files are already committed.")
            return

        filelist = []
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith(".xml"):
                self.logger.info(
                    "Moving {0} to data/committedframes/{0}".format(filename))
                filename = os.path.join(path, filename)
                basename = os.path.splitext(filename)[0]
                filelist.append(filename)
                filelist.append(basename + '.jpg')
            else:
                continue

        for i in filelist:
            dest = os.path.join(self.committedframesDataPath, os.path.split(i)[1])
            try:
                os.rename(i, dest)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    shutil.copy2(i, dest)
                    os.remove(i)
                else:
                    raise

        self.importDirImages(path)
