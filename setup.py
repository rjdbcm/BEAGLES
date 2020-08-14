from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from urllib.parse import urljoin
import numpy
import sys
import os
from libs.constants import APP_NAME
from libs.version import __version__

GITHUB = urljoin('https://github.com/rjdbcm/', APP_NAME)


def module_to_path(module):
    module = module.replace('.', '/')
    module += '.pyx'
    return [module]


NMS = 'libs.cythonUtils.nms'
CY_YOLO_FINDBOXES = 'libs.cythonUtils.cy_yolo_findboxes'
CY_YOLO2_FINDBOXES = 'libs.cythonUtils.cy_yolo2_findboxes'
WINDOWS = 'nt'
MAC = 'darwin'


with open("README.md", "r") as f:
    long_description = f.read()

# noinspection SpellCheckingInspection
short_description = 'BEhavioral Annotation and Gesture LEarning Suite'

DESCRIPTION_MIME = 'text/markdown'

if os.name == WINDOWS:
    ext_modules = [
        Extension(NMS,
                  sources=module_to_path(NMS),
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension(CY_YOLO2_FINDBOXES,
                  sources=module_to_path(CY_YOLO2_FINDBOXES),
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['/fopenmp'],
                  extra_link_args=['/fopenmp']
                  ),
        Extension(CY_YOLO_FINDBOXES,
                  sources=module_to_path(CY_YOLO_FINDBOXES),
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  )
    ]
elif os.name == 'posix':
    if sys.platform == MAC:
        compile_args = ''
        linker_args = ''
    else:
        # This gives a significant boost to postprocessing time
        compile_args = ['-fopenmp', '-funroll-loops']
        linker_args = ['-fopenmp']

    ext_modules = [
        Extension(NMS,
                  sources=module_to_path(NMS),
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension(CY_YOLO2_FINDBOXES,
                  sources=module_to_path(CY_YOLO2_FINDBOXES),
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=compile_args,
                  extra_link_args=linker_args
                  ),
        Extension(CY_YOLO_FINDBOXES,
                  sources=module_to_path(CY_YOLO_FINDBOXES),
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  )
    ]
else:
    ext_modules = [
        Extension(NMS,
                  sources=module_to_path(NMS),
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension(CY_YOLO2_FINDBOXES,
                  sources=module_to_path(CY_YOLO2_FINDBOXES),
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension(CY_YOLO_FINDBOXES,
                  sources=module_to_path(CY_YOLO_FINDBOXES),
                  libraries=["m"]  # Unix-like specific
                  )
    ]

setup(
    version=__version__,
    name=APP_NAME,
    description=short_description,
    description_content_type=DESCRIPTION_MIME,
    long_description=long_description,
    long_description_content_type=DESCRIPTION_MIME,
    license='GPLv3',
    url=GITHUB,
    packages=find_packages(),
    scripts=['BEAGLES.py'],
    ext_modules=cythonize(ext_modules),
    extras_require={
        'darkmode': ["qdarkstyle", "pyobjc"] if sys.platform == MAC else ["qdarkstyle"],
        'dev': ["googletrans"]
    },
    classifiers=["Programming Language :: Cython",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Topic :: Scientific/Engineering :: Bio-Informatics",
                 "Intended Audience :: Science/Research",
                 "Development Status :: 2 - Pre-Alpha"],
    install_requires=["PyQt5",
                      "defusedxml>=0.6.0",
                      "lxml>=4.2.4",
                      "Cython>=0.29.6",
                      "opencv-contrib-python==4.0.0.21",
                      "tensorflow>2.0.0b",
                      "numpy>=1.16.2",
                      "traces>=0.4.2"]

)