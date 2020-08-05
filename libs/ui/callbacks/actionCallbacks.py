from libs.ui.callbacks.editCallbacks import EditCallbacks
from libs.ui.callbacks.fileCallbacks import FileCallbacks
from libs.ui.callbacks.helpCallbacks import HelpCallbacks
from libs.ui.callbacks.machineLearningCallbacks import MachineLearningCallbacks
from libs.ui.callbacks.viewCallbacks import ViewCallbacks


class ActionCallbacks(FileCallbacks, ViewCallbacks, EditCallbacks,
                      HelpCallbacks, MachineLearningCallbacks):
    """Interface class for menu action callbacks"""
    def __init__(self):
        super(ActionCallbacks, self).__init__()
