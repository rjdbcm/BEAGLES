#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from libs.version import __version__
from sys import platform as _platform
from Cython.Build import cythonize
import numpy
import imp
import os

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    # TODO: Different OS have different requirements
]

# OS specific settings
SET_REQUIRES = []
if _platform == "linux" or _platform == "linux2":
   # linux
   print('linux')
elif _platform == "darwin":
   # MAC OS X
   SET_REQUIRES.append('py2app')

required_packages = find_packages()
required_packages.append('slgrSuite')

APP = ['slgrSuite.py']
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'resources/icons/app.icns',
    'packages': ['certifi']
}


setup(
    app=APP,
    name='SLGR-Suite',
    version=__version__,
    description="SLGR-Suite is a graphical image annotation tool and frontend for machine learning algorithms",
    long_description=readme + '\n\n' + history,
    author="Ross J. Duff",
    author_email='rjdbcm@mail.umkc.edu',
    url='https://github.com/rjdbcm/slgr-suite',
    package_dir={'slgrSuite': '.'},
    packages=required_packages,
    entry_points={
        'console_scripts': [
            'SLGR-Suite=slgrSuite.slgrSuite:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='YOLO development annotation deeplearning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    package_data={'data/predefined_classes.txt': ['data/predefined_classes.txt']},
    options={'py2app': OPTIONS},
    setup_requires=SET_REQUIRES
)
