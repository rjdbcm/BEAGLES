#!/bin/sh
# To be run from the build-tools/ directory
# TODO: Install into a virtualenv so the binary isn't massive
# Run this script as the super user after running:

which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

which -s python3
if [[ $? != 0 ]] ; then
    # Install Python3
    brew install python@3
fi

# clean out any old build files
cd ../
rm -rf build
rm -rf dist

# build SLGR-Suite
pip3 install pyinstaller opencv-contrib-python-headless PyQt5 lxml tensorflow numpy Cython
make qt5py3
pyinstaller -w --add-data ./data:data --add-data ./backend:backend --icon=resources/icons/app.icns -n SLGR-Suite slgrSuite.py
mv "dist/SLGR-Suite.app" /Applications

# symlink the backend and data folders
mkdir -p ~/SLGR-Suite/backend
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/flow /usr/local/bin/flow
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/bin/ ~/SLGR-Suite/backend/bin
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/ckpt/ ~/SLGR-Suite/backend/ckpt
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/built_graph/ ~/SLGR-Suite/backend/built_graph
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/sample_img/ ~/SLGR-Suite/backend/sample_img
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/cfg/ ~/SLGR-Suite/backend/cfg
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/ ~/SLGR-Suite/data

echo 'DONE'