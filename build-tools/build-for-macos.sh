#!/bin/sh
# To be run from the build-tools/ directory

brew install python@3

# clean out any old build files
cd ../
rm -rf build
rm -rf dist

# build SLGR-Suite
pip install pyinstaller opencv-contrib-python-headless PyQt5 lxml tensorflow numpy
make qt5py3
pyinstaller -w slgrSuite.spec
mv "dist/SLGR-Suite.app" /Applications

echo 'DONE'