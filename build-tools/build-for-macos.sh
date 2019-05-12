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
pip install pyinstaller opencv-contrib-python-headless PyQt5 lxml tensorflow numpy
make qt5py3
pyinstaller -w slgrSuite.spec
mv "dist/SLGR-Suite.app" /Applications

# symlink the backend and data folders
mkdir -p ~/SLGR-Suite
sudo ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/flow /usr/local/bin/flow
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/bin/ ~/SLGR-Suite/bin
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/ckpt/ ~/SLGR-Suite/ckpt
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/built_graph/ ~/SLGR-Suite/built_graph
ln -s /Applications/SLGR-Suite.app/Contents/Resources/backend/cfg/ ~/SLGR-Suite/cfg
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/ ~/SLGR-Suite/data

echo 'DONE'