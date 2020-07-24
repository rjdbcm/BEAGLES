#!/bin/sh
# To be run from the build-tools/ directory
# These scripts are here for those interested in attempting a binary build.
# It NOT RECOMMENDED, please use build-venv.sh

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

# build BEAGLES
pip3 install pyinstaller
pip3 install -r requirements/requirements.txt
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
            -n BEAGLES slgrSuite.py
mv "dist/BEAGLES.app" /Applications

# symlink the data folders
mkdir ~/BEAGLES
ln -s /Applications/BEAGLES.app/Contents/Resources/data/bin/ ~/BEAGLES/bin
ln -s /Applications/BEAGLES.app/Contents/Resources/data/logs/ ~/BEAGLES/logs
ln -s /Applications/BEAGLES.app/Contents/Resources/data/summaries/ ~/BEAGLES/summaries
ln -s /Applications/BEAGLES.app/Contents/Resources/data/ckpt/ ~/BEAGLES/ckpt
ln -s /Applications/BEAGLES.app/Contents/Resources/data/built_graph/ ~/BEAGLES/built_graph
ln -s /Applications/BEAGLES.app/Contents/Resources/data/sample_img/ ~/BEAGLES/sample_img
ln -s /Applications/BEAGLES.app/Contents/Resources/data/sample_img/out/ ~/BEAGLES/sample_img/out
ln -s /Applications/BEAGLES.app/Contents/Resources/data/cfg/ ~/BEAGLES/cfg
ln -s /Applications/BEAGLES.app/Contents/Resources/data/predefined_classes.txt ~/BEAGLES/data/predefined_classes.txt

echo 'DONE'