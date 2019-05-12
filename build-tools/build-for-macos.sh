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

# symlink the backend and data folders
mkdir -p ~/SLGR-Suite
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/flow /usr/sbin/flow
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/bin/ ~/SLGR-Suite/bin/
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/ckpt/ ~/SLGR-Suite/ckpt/
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/built_graph/ ~/SLGR-Suite/built_graph/
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/cfg/ ~/SLGR-Suite/cfg/
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/ ~/SLGR-Suite/data/

echo 'DONE'