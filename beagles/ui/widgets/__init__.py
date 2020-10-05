"""QWidget children used the BEAGLES UI"""
from beagles.ui.widgets.beaglesMainWindow import BeaglesMainWindow
from beagles.ui.widgets.backend import BackendThread, BackendConnection, BackendDialog
from beagles.ui.widgets.canvasWidget import CanvasWidget
from beagles.ui.widgets.colorDialog import ColorDialog
from beagles.ui.widgets.floatQValidator import FloatQValidator, FLOAT_RE
from beagles.ui.widgets.flowDialog import FlowDialog
from beagles.ui.widgets.hashableQListWidgetItem import HashableQListWidgetItem
from beagles.ui.widgets.labelDialog import LabelDialog, LABEL_RE
from beagles.ui.widgets.projectDialog import ProjectDialog
from beagles.ui.widgets.scientificQDoubleSpinBox import ScientificQDoubleSpinBox
from beagles.ui.widgets.toolBar import ToolBar, ToolButton
from beagles.ui.widgets.zoomWidget import ZoomWidget

BeaglesMainWindow = BeaglesMainWindow
"""
:class:`beagles.ui.widgets.beaglesMainWindow.BeaglesMainWindow`
"""

BackendThread = BackendThread
"""
:class:`beagles.ui.widgets.backend.BackendThread`
"""

BackendConnection = BackendConnection
"""
:class:`beagles.ui.widgets.backend.BackendConnection`
"""

LabelDialog = LabelDialog
"""
:class:`beagles.ui.widgets.labelDialog.LabelDialog`
"""