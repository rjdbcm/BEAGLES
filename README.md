# SLGR-Suite


SLGR-Suite is a graphical image annotation tool.

It is forked from LabelImg, written in Python, and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
also supports YOLO format

## Installation



### Download prebuilt binaries


-  Binaries are not yet available but the build isn't *too* hard.

### Build from source

Linux/Ubuntu/Mac requires at least [Python
3.6](https://www.python.org/getit/) and has been tested with [PyQt
5.10.1](https://www.riverbankcomputing.com/software/pyqt/intro)


### Ubuntu Linux

* Python 3 + Qt5 (Recommended)

```bash
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
### macOS


* Python 3 + Qt5 (Recommended)

```bash
   
    brew install qt  # Install qt-5.x.x by Homebrew
    brew install libxml2

    or

    pip3 install pyqt5 lxml # Install qt and lxml by pip
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

```
### Python 3 Virtualenv + Binary

This can avoid a lot of the QT / Python version issues, and gives you a nice .app file with a new SVG Icon
in your /Applications folder. You can consider this script: build-tools/build-for-macos.sh

```bash

    brew install python3
    pip install pipenv
    pipenv --three
    pipenv shell
    pip install py2app
    pip install PyQt5 lxml
    make qt5py3
    rm -rf build dist
    python setup.py py2app -A
    mv "dist/SLGR-Suite.app" /Applications
```

###Windows


Install:
* [Python](https://www.python.org/downloads/windows/)
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5)
* [lxml](http://lxml.de/installation.html)

Open cmd and go to the `slgr_suite` directory
```bash

    pyrcc4 -o resources.py resources.qrc
    python labelImg.py
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

###Windows + Anaconda


Download and install [Anaconda (Python 3+)](https://www.anaconda.com/download/#download) 

Open the Anaconda Prompt and go to the `slgr_suite` directory

```bash
    conda install pyqt=5
    pyrcc5 -o resources.py resources.qrc
    python labelImg.py
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

## Usage

### Create pre-defined classes

Edit the
[data/predefined_classes.txt](https://github.com/rjdbcm/slgr-suite/blob/master/data/predefined_classes.txt)
to load pre-defined classes of your own.

###Steps (PascalVOC)


1. Build and launch using the instructions above.
2. Click 'Change default saved annotation folder' in Menu/File
3. Click 'Open Dir'
4. Click 'Create RectBox'
5. Click and release left mouse to select a region to annotate the rect
   box
6. You can use right mouse to drag the rect box to copy or move it

The annotation will be saved to the folder you specify.

You can refer to the below hotkeys to speed up your workflow.

###Steps (YOLO)


1. In ``data/predefined_classes.txt`` define the list of classes that will be used for your training.

2. Build and launch using the instructions above.

3. Right below "Save" button in toolbar, click "PascalVOC" button to switch to YOLO format.

4. You may use Open/OpenDIR to process single or multiple images. When finished with single image, click save.

A txt file of yolo format will be saved in the same folder as your image with same name. A file named "classes.txt" is saved to that folder too. "classes.txt" defines the list of class names that your yolo label refers to.

Note:

- Your label list shall not change in the middle of processing a list of images. When you save a image, classes.txt will also get updated, while previous annotations will not be updated.

- You shouldn't use "default class" function when saving to YOLO format, it will not be referred.

- When saving as YOLO format, "difficult" flag is discarded.

## Controls
|  Hotkey  |                     Action                     |
|:--------:|:----------------------------------------------:|
| Ctrl ⇧ a | Toggle advanced mode toolbar                   |
| Ctrl +   | Zoom in                                        |
| Ctrl -   | Zoom out                                       |
| Ctrl i   | Choose a video to import frames for annotation |
| Ctrl u   | Choose a directory to load images from         |
| Ctrl r   | Change the default annotation directory        |
| Ctrl s   | Save                                           |
| Crtl d   | Copy the selected label and bounding box       |
| Space    | Flag the current image as verified             |
| w        | Create a new bounding box                      |
| d        | Next image                                     |
| s        | Previous Image                                 |
| del      | Delete the selected bounding box               |
| ↑→↓←     | Move the selected bounding box                 |

### Verify Image

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when creating a dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

### Difficult

The difficult field being set to 1 indicates that the object has been annotated as "difficult", for example an object which is clearly visible but difficult to recognize without substantial use of context.
According to your deep neural network implementation, you can include or exclude difficult objects during training.



How to contribute
-----------------

Send a pull request

License
-------
Free software: [MIT license](https://github.com/rjdbcm/slgr-suite/blob/master/LICENSE)

Based on original code by: Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg

Related
-------

1. [labelImg](https://github.com/tzutalin/labelImg) the original image annotation software SLGR-Suite is forked from
2. [darkflow](https://github.com/thtrieu/darkflow) the original basis of the machine learning backend

