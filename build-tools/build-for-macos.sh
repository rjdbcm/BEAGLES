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

rm -rf /tmp/SLGR-SuiteSetup
mkdir /tmp/SLGR-SuiteSetup
cd /tmp/SLGR-SuiteSetup
curl https://codeload.github.com/rjdbcm/slgrSuite/zip/master --output slgrSuite.zip
unzip slgrSuite.zip
rm slgrSuite.zip

# clean out any old build files
rm -rf build
rm -rf dist

virtualenv -p python3 /tmp/SLGR-SuiteSetup/slgrSuite-py3
source /tmp/SLGR-SuiteSetup/slgrSuite-py3/bin/activate
cd slgrSuite-master

# build SLGR-Suite
pip install pyinstaller opencv-contrib-python-headless PyQt5 lxml tensorflow numpy Cython
make qt5py3
pyinstaller -w slgrSuite.spec
mv "dist/SLGR-Suite.app" /Applications
deactivate

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