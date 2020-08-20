from os.path import expanduser, join
from os import getcwd
from datetime import datetime

APP_NAME = 'BEAGLES'
SAVE_SETTINGS_PATH = join(expanduser("~"), '.BEAGLESSettings.json')
DEFAULT_FLAGS_PATH = join(getcwd(), 'resources/flags.json')
BACKEND_ENTRYPOINT = join(getcwd(), 'libs/backend/wrapper.py')
EPOCH = datetime(1970, 1, 1, 0, 0)
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
FLOAT_RE = r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)'
LABEL_RE = r'^[^ \t].+'
