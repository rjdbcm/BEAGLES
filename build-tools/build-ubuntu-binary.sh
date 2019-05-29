#!/bin/sh
### Not current TODO: Update this

cd ..
rm -r build
rm -r dist
rm labelImg.spec
pip3 install pyinstaller
pip3 install -r requirements/requirements-linux.txt
make all
pyinstaller --hidden-import=xml \
            --hidden-import=xml.etree \
            --hidden-import=xml.etree.ElementTree \
            --hidden-import=lxml.etree \
            -r libs/cython_utils/cy_yolo_findboxes.so \
            -r libs/cython_utils/cy_yolo2_findboxes.so \
            -r libs/cython_utils/nms.so \
            --add-data ./data:data \
            --icon=resources/icons/app.icns \
            -n SLGR-Suite slgrSuite.py \
            -D -p ./libs -p ./

FOLDER=$(git describe --abbrev=0 --tags)
FOLDER="linux_"$FOLDER
rm -rf "$FOLDER"
mkdir "$FOLDER"
cp dist/SLGR-Suite $FOLDER
cp -rf ./data $FOLDER/data
zip "$FOLDER.zip" -r $FOLDER
