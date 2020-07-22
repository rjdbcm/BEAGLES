# BEAGLES
BEAGLES stands for **BE**havioral **A**nnotation and **G**esture **LE**arning **S**uite, and is intended for behavioral analysis and quantification. BEAGLES is a graphical image annotation 
tool originally forked from [labelImg](https://github.com/tzutalin/labelImg) and frontend for a fork of 
[darkflow](https://github.com/thtrieu/darkflow). 

Written in Python, SLGR-Suite uses Qt for its graphical interface.
By default annotations are saved as ```.xml``` files in PASCAL VOC format but there is also support for saving YOLO 
formatted ```.txt``` files.

## Installation

### Source Install (virtualenv)

Navigate to the source directory and run the following commands:

```
cd build/
./build-venv.sh
cd ..
make
```

### Binary Build

Scripts are included in `build/` for those interested but are **NOT** **RECOMMENDED**.

## Usage

From the source directory run:
```
./slgrSuite.py
```

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
Free software:
- [GPLv3](https://github.com/rjdbcm/slgrSuite/blob/master/LICENSE)
- [MIT NOTICE](https://github.com/rjdbcm/slgrSuite/blob/master/NOTICE)

Based in part on original code by: 
- Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg
- Mahmoud Aslan. Cyclic Learning Rate. Git code (2018). https://github.com/mhmoodlan/cyclic-learning-rate


Related
-------

1. [labelImg](https://github.com/tzutalin/labelImg) the original image annotation software SLGR-Suite is forked from
2. [darkflow](https://github.com/thtrieu/darkflow) the original basis of the machine learning backend
3. [cyclic-learning-rate](https://github.com/mhmoodlan/cyclic-learning-rate) the implementation of cyclic learning rates used
