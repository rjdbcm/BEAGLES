#!/bin/sh
# TODO: Switch this all to PyInstaller or fman since they actually work unlike py2app

brew install python@3
pip install --upgrade virtualenv 

# clone source
rm -rf /tmp/SLGR-SuiteSetup
mkdir /tmp/SLGR-SuiteSetup
cd /tmp/SLGR-SuiteSetup
curl https://codeload.github.com/rjdbcm/slgrSuite/zip/master --output slgrSuite.zip
unzip slgrSuite.zip
rm slgrSuite.zip

# setup python3 virtualenv
virtualenv --system-site-packages  -p python3 /tmp/SLGR-SuiteSetup/slgrSuite-py3
source /tmp/SLGR-SuiteSetup/slgrSuite-py3/bin/activate
cd slgrSuite-master

# build labelImg app
pip install pyinstaller
pip install PyQt5 lxml tensorflow opencv-python-headless numpy sip
make qt5py3
pyinstaller -w slgrSuite.spec
mv "/tmp/SLGR-SuiteSetup/slgrSuite-master/dist/SLGR-Suite.app" /Applications

# deactivate python3 virtualenv
deactivate
cd ../
rm -rf /tmp/SLGR-SuiteSetup
echo 'DONE'