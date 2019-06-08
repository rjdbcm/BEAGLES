#!/bin/sh
# To be run from the build-tools/ directory

which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

if brew ls --versions libxml2 > /dev/null; then
    # Install libxml
    brew install libxml2
fi

if brew ls --versions qt > /dev/null; then
    # Install qt
    brew install qt
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
pip3 install pyinstaller
pip3 install -r requirements/requirements-osx-mojave.txt
make qt5py3
pyinstaller -w --hidden-import=xml \
            --hidden-import=xml.etree \
            --hidden-import=xml.etree.ElementTree \
            --hidden-import=lxml.etree \
            -r libs/cython_utils/cy_yolo_findboxes.so \
            -r libs/cython_utils/cy_yolo2_findboxes.so \
            -r libs/cython_utils/nms.so \
            --add-data ./data:data \
            --icon=resources/icons/app.icns \
            -n SLGR-Suite slgrSuite.py
mv "dist/SLGR-Suite.app" /Applications

# symlink the data folders
mkdir ~/SLGR-Suite
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/bin/ ~/SLGR-Suite/bin
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/logs/ ~/SLGR-Suite/logs
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/summaries/ ~/SLGR-Suite/summaries
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/ckpt/ ~/SLGR-Suite/ckpt
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/built_graph/ ~/SLGR-Suite/built_graph
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/sample_img/ ~/SLGR-Suite/sample_img
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/sample_img/out/ ~/SLGR-Suite/sample_img/out
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/cfg/ ~/SLGR-Suite/cfg
ln -s /Applications/SLGR-Suite.app/Contents/Resources/data/predefined_classes.txt ~/SLGR-Suite/data/predefined_classes.txt

echo 'DONE'