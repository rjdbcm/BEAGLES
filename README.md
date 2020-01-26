# SLGR-Suite
SLGR stands for Single-Look Gesture Recognition and is an implementation of the YOLO(You Only Look Once) object 
classification system extended for behavioral analysis and quantification. SLGR-Suite is a graphical image annotation 
tool forked from [labelImg](https://github.com/tzutalin/labelImg) and frontend for a custom fork of 
[darkflow](https://github.com/thtrieu/darkflow). 

Written in Python, SLGR-Suite uses Qt for its graphical interface.
By default annotations are saved as ```.xml``` files in PASCAL VOC format but there is also support for saving YOLO 
formatted ```.txt``` files.

## Installation

### Binary Build Requirements
#### Ubuntu Linux

* libomp-dev 

* pyqt5-dev-tools

* python3-pip

* Python 3.6 or above

* The following PyPi packages from ```requirements/requirements-linux.txt```:
```python
pyqt5>=5.10.1
lxml==4.2.4
Cython==0.29.6
opencv-contrib-python==4.0.0.21
tensorflow==1.13.1
numpy==1.16.2
traces==0.4.2
```
##### Build Binary:
```bash
cd build_tools
./build_ubuntu_binary.sh
```

#### MacOS
* [Homebrew](https://brew.sh)

* qt

* libxml2

* Python 3.6 or above

* The following PyPi packages from ```requirements/requirements-osx-mojave.txt```
```python
pyqt5>=5.12
lxml==4.2.4
Cython==0.29.6
opencv-contrib-python-headless==4.0.0.21
tensorflow==1.13.1
numpy==1.16.2
```
##### Build Binary: (NOT WORKING)
```bash
cd build_tools
./build_for_macos.sh
```

### Download prebuilt binaries

-  Binaries are not yet available but the build isn't hard.

### Build from and run as a source distribution

- Linux/Ubuntu has been tested with [Python 3.6](https://www.python.org/getit/) & [PyQt 5.10.1](https://www.riverbankcomputing.com/software/pyqt/intro)

- MacOS has been tested with [Python
3.7](https://www.python.org/getit/) & [PyQt
5.12](https://www.riverbankcomputing.com/software/pyqt/intro) installed using [homebrew](https://brew.sh).


#### Ubuntu Linux

* Run as a source distribution:

```bash
sudo apt-get install python3-pip
sudo apt-get install pyqt5-dev-tools
sudo apt-get install libomp-dev
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 slgrSuite.py
python3 slgrSuite.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

#### macOS


* Python 3 + Qt5 (Recommended)

```bash
brew install qt  # Install qt-5.x.x by Homebrew
brew install libxml2
pip3 install -r requirements/requirements-osx-mojave.txt
make qt5py3
python3 slgrSuite.py
python3 slgrSuite.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

## Usage

### Create pre-defined classes

Edit the
[data/predefined_classes.txt](https://github.com/rjdbcm/slgr-suite/blob/master/data/predefined_classes.txt)
to load pre-defined classes of your own.

### Steps

1. In `data/predefined_classes.txt` define the list of classes that will be used for your training.

2. Build and launch using the instructions above.

3. Right next to the "Save" button in toolbar you can choose PascalVOC `.xml` or YOLO `.txt` the default is PascalVOC.

4. You may use Open/Open Folder to process single or multiple images. You may also use Import Video Frames from advanced mode to import every frame from a selected video file into `data/rawframes`. When finished with single image, click save or you can activate autosave mode.

    A `.txt` or `.xml` file of the annotations will be saved in the same folder as your image with same name. A file named "classes.txt" is saved to that folder too. "classes.txt" defines the list of class names that your yolo label refers to.

5. When finished annotating commit the images and corresponding annotation files to the `data/committedframes` folder by pressing the Commit Frames button on the advanced mode toolbar.

6. (OPTIONAL) Download [pretrained weights](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) for the model configurations provided by [thtrieu](https://github.com/thtrieu) which include yolo-full and yolo-tiny of v1.0, tiny-yolo-v1.1 of v1.1 and yolo, tiny-yolo-voc of v2.

7. Press Ctrl+T (&#8984;T) to start the machine learning interface.

8. Enter parameters and start training a model.

9. Annotate video files using your trained model.

*Important Notes:*

- Your label list shall not change in the middle of processing a list of images, this includes the order. When you save a image, classes.txt will also get updated, while previous annotations will not be updated.

- You shouldn't use "default class" function when saving to YOLO format, it will not be referred.

- When saving as YOLO format, "difficult" flag is discarded.

- The PascalVOC annotation `.xml` output from SLGR-Suite does not include the `<path>` element.

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
| Crtl d   | Duplicate the selected label and bounding box  |
| Ctrl t   | Open machine learning interface                |
| Space    | Flag the current image as verified             |
| w        | Create a new bounding box                      |
| d        | Next image                                     |
| s        | Previous Image                                 |
| del      | Delete the selected bounding box               |
| ↑→↓←     | Move the selected bounding box                 |
|:--------:|:----------------------------------------------:|

### Verify Image

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when creating a dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

### Difficult

The difficult field being set to 1 indicates that the object has been annotated as "difficult", for example an object which is clearly visible but difficult to recognize without substantial use of context.
According to your deep neural network implementation, you can include or exclude difficult objects during training.

## Important Directories
* Frames are imported to a folder named for the video filename in ```data/rawframes```.

* When you press Commit Frames images in the open directory with matching annotation files are moved into ```data/committedframes```.

* Tensorboard summaries are found in```data/summaries```

* Training checkpoints are saved in```data/ckpt```

* Frozen graph files (*.pb, *.meta) output in```data/built_graph```

* Model configurations are stored in```data/cfg```

* Pretrained weights should be saved into```data/bin```

* Text-based log files are in ```data/logs```

* Image sets to annotate are stored in ```data/sample_img```


How to contribute
-----------------

Send a pull request

License
-------
Free software: [GPLv3](https://github.com/rjdbcm/slgrSuite/blob/master/LICENSE)

[NOTICE](https://github.com/rjdbcm/slgrSuite/blob/master/NOTICE):
Based on original code by: Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg

Related
-------

1. [labelImg](https://github.com/tzutalin/labelImg) the original image annotation software SLGR-Suite is forked from
2. [darkflow](https://github.com/thtrieu/darkflow) the original basis of the machine learning backend

