from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import imp

VERSION = imp.load_source('version', os.path.join('.', 'libs', 'version.py'))
VERSION = VERSION.__version__

with open("README.md", "r") as fh:
    long_description = fh.read()

if os.name == 'nt':
    ext_modules = [
        Extension("libs.cython_utils.nms",
                  sources=["libs/cython_utils/nms.pyx"],
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("libs.cython_utils.cy_yolo2_findboxes",
                  sources=["libs/cython_utils/cy_yolo2_findboxes.pyx"],
                  # libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['/fopenmp'],
                  extra_link_args=['/fopenmp']
                  ),
        Extension("libs.cython_utils.cy_yolo_findboxes",
                  sources=["libs/cython_utils/cy_yolo_findboxes.pyx"],
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
        Extension("libs.cython_utils.nms",
                  sources=["libs/cython_utils/nms.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("libs.cython_utils.cy_yolo2_findboxes",
                  sources=["libs/cython_utils/cy_yolo2_findboxes.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=compile_args,
                  extra_link_args=linker_args
                  ),
        Extension("libs.cython_utils.cy_yolo_findboxes",
                  sources=["libs/cython_utils/cy_yolo_findboxes.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()]
                  )
    ]
else:
    ext_modules = [
        Extension("libs.cython_utils.nms",
                  sources=["libs/cython_utils/nms.pyx"],
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension("libs.cython_utils.cy_yolo2_findboxes",
                  sources=["libs/cython_utils/cy_yolo24findboxes.pyx"],
                  libraries=["m"]  # Unix-like specific
                  ),
        Extension("libs.cython_utils.cy_yolo_findboxes",
                  sources=["libs/cython_utils/cy_yolo_findboxes.pyx"],
                  libraries=["m"]  # Unix-like specific
                  )
    ]

setup(
    version=VERSION,
    name='SLGR-Suite',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url='https://github.com/rjdbcm/SLGR-Suite',
    packages=find_packages(),
    scripts=['slgrSuite.py'],
    ext_modules=cythonize(ext_modules),
    extras_require={'darkmode': ["qdarkstyle", "pyobjc"]
                    if sys.platform == "Darwin" else ["qdarkstyle"]},
    classifiers=["Programming Language :: Cython",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved ::"
                 " GNU General Public License v3 (GPLv3)",
                 "Topic :: Scientific/Engineering :: Bio-Informatics",
                 "Intended Audience :: Science/Research",
                 "Development Status :: 2 - Pre-Alpha"],
)
