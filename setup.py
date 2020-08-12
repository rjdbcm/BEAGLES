from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import importlib

VERSION = importlib.import_module('.version', 'libs')
VERSION = VERSION.__version__
PLAT = sys.platform

with open("README.md", "r") as f:
    long_description = f.read()

if os.name == 'nt':
    ext_modules = [
        Extension("libs.cythonUtils.nms",
                  sources=["libs/cythonUtils/nms.pyx"],
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("libs.cythonUtils.cy_yolo2_findboxes",
                  sources=["libs/cythonUtils/cy_yolo2_findboxes.pyx"],
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['/fopenmp'],
                  extra_link_args=['/fopenmp']
                  ),
        Extension("libs.cythonUtils.cy_yolo_findboxes",
                  sources=["libs/cythonUtils/cy_yolo_findboxes.pyx"],
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  )
    ]
elif os.name == 'posix':
    if sys.platform == 'darwin':
        compile_args = ''
        linker_args = ''
    else:
        # This gives a significant boost to postprocessing time
        compile_args = ['-fopenmp', '-funroll-loops']
        linker_args = ['-fopenmp']

    ext_modules = [
        Extension("libs.cythonUtils.nms",
                  sources=["libs/cythonUtils/nms.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("libs.cythonUtils.cy_yolo2_findboxes",
                  sources=["libs/cythonUtils/cy_yolo2_findboxes.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=compile_args,
                  extra_link_args=linker_args
                  ),
        Extension("libs.cythonUtils.cy_yolo_findboxes",
                  sources=["libs/cythonUtils/cy_yolo_findboxes.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  )
    ]
else:
    ext_modules = [
        Extension("libs.cythonUtils.nms",
                  sources=["libs/cythonUtils/nms.pyx"],
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension("libs.cythonUtils.cy_yolo2_findboxes",
                  sources=["libs/cythonUtils/cy_yolo2_findboxes.pyx"],
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension("libs.backend.cythonUtils.cy_yolo_findboxes",
                  sources=["libs/backend/cythonUtils/cy_yolo_findboxes.pyx"],
                  libraries=["m"]  # Unix-like specific
                  )
    ]

setup(
    version=VERSION,
    name='BEAGLES',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url='https://github.com/rjdbcm/BEAGLES',
    packages=find_packages(),
    scripts=['BEAGLES.py'],
    ext_modules=cythonize(ext_modules),
    extras_require={'darkmode': ["qdarkstyle", "pyobjc"] if PLAT == "Darwin" else ["qdarkstyle"],
                    'dev': ["googletrans"]},
    classifiers=["Programming Language :: Cython",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved ::"
                 " GNU General Public License v3 (GPLv3)",
                 "Topic :: Scientific/Engineering :: Bio-Informatics",
                 "Intended Audience :: Science/Research",
                 "Development Status :: 2 - Pre-Alpha"],
    install_requires=["PyQt5",
                      "defusedxml>=0.6.0",
                      "lxml>=4.2.4",
                      "Cython==0.29.6",
                      "opencv-contrib-python>=4.0.0.21",
                      "tensorflow>2.0.0b",
                      "numpy==1.16.2",
                      "traces==0.4.2"]

)