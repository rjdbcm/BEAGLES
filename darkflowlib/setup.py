from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import imp

VERSION = imp.load_source('version', os.path.join('.', 'darkflow', 'version.py'))
VERSION = VERSION.__version__



if os.name =='nt' :
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["darkflow/cython_utils/nms.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
            sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()],
            extra_compile_args=['/fopenmp'],
            extra_link_args=['/fopenmp']
        ),
        Extension("darkflow.cython_utils.cy_yolo_findboxes",
            sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

elif os.name =='posix' :
    if sys.platform == 'darwin':
        compile_args = ''
        linker_args = ''
    else:
        compile_args = '-fopenmp'
        linker_args = '-fopenmp'
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["darkflow/cython_utils/nms.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
                  sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
                  libraries=["m"],  # Unix-like specific
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=[compile_args],
                  extra_link_args=[linker_args]
                  ),
        Extension("darkflow.cython_utils.cy_yolo_findboxes",
            sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

else :
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["darkflow/cython_utils/nms.pyx"],
            libraries=["m"] # Unix-like specific
        ),        
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
            sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        ),
        Extension("darkflow.cython_utils.cy_yolo_findboxes",
            sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        )
    ]

setup(
    version=VERSION,
	name='darkflow',
    description='Darkflow',
    license='GPLv3',
    url='https://github.com/rjdbcm/darkflow',
    packages = find_packages(),
	scripts = ['flow'],
    ext_modules = cythonize(ext_modules)
)