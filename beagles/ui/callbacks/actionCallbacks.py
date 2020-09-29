from beagles.ui.callbacks.editCallbacks import EditCallbacks
from beagles.ui.callbacks.fileCallbacks import FileCallbacks
from beagles.ui.callbacks.helpCallbacks import HelpCallbacks
from beagles.ui.callbacks.machineLearningCallbacks import MachineLearningCallbacks
from beagles.ui.callbacks.viewCallbacks import ViewCallbacks


class ActionCallbacks(FileCallbacks, ViewCallbacks, EditCallbacks,
                      HelpCallbacks, MachineLearningCallbacks):
    """Interface class for menu action callbacks"""
    def __init__(self):
        super(ActionCallbacks, self).__init__()
