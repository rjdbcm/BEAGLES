from os.path import expanduser, join
from os import getcwd
from datetime import datetime
import re
import logging

INFO = logging.INFO
WARN = logging.WARN
DEBUG = logging.DEBUG
ERROR = logging.ERROR
APP_NAME = 'BEAGLES'
FLAG_FILE = ".flags.json"
SHELL_HISTORY_PATH = expanduser('~/.beagles_history')
SHELL_HISTORY_SIZE = 1000
SAVE_SETTINGS_PATH = expanduser('~/.BEAGLESSettings.json')
DEFAULT_FLAGS_PATH = join(getcwd(), 'resources/flags.json')
BACKEND_ENTRYPOINT = 'beagles/backend/wrapper.py'
EPOCH = datetime.fromtimestamp(0)
SETTING_FILENAME = 'filename'
SETTING_RECENT_FILES = 'recentFiles'
SETTING_WIN_SIZE = 'window/size'
SETTING_WIN_POSE = 'window/position'
SETTING_WIN_GEOMETRY = 'window/geometry'
SETTING_LINE_COLOR = 'line/color'
SETTING_FILL_COLOR = 'fill/color'
SETTING_ADVANCE_MODE = 'advanced'
SETTING_WIN_STATE = 'window/state'
SETTING_SAVE_DIR = 'savedir'
SETTING_PAINT_LABEL = 'paintlabel'
SETTING_LAST_OPEN_DIR = 'lastOpenDir'
SETTING_AUTO_SAVE = 'autosave'
SETTING_SINGLE_CLASS = 'singleclass'
FORMAT_PASCALVOC = 'PascalVOC'
FORMAT_YOLO = 'YOLO'
SETTING_DRAW_SQUARE = 'draw/square'
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
TXT_EXT = '.txt'
CFG_EXT = '.cfg'
WGT_EXT = '.weights'
FLOAT_RE = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')
LABEL_RE = re.compile(r'^[^ \t].+')

WEIGHTS_FILE_KEYS = dict({  # order of param flattened into .weights file
    'convolutional': [
        'biases',
        'gamma',
        'moving_mean',
        'moving_variance',
        'kernel'
    ],
    'connected': [
        'biases',
        'weights'
    ],
    'local': [
        'biases',
        'kernels'
    ]
})
