# SLGR-Suite
SLGR stands for Single-Look Gesture Recognition and is an implementation of the YOLO(You Only Look Once) object 
classification system extended for behavioral analysis and quantification. SLGR-Suite is a graphical image annotation 
tool originally forked from [labelImg](https://github.com/tzutalin/labelImg) and frontend for a fork of 
[darkflow](https://github.com/thtrieu/darkflow). 

Written in Python, SLGR-Suite uses Qt for its graphical interface and TensorFlow for its backend.

##### *Known Issues:*

- Cython extensions leak memory in some cases

##### Features:

- Darknet-style configuration files 
- TensorFlow checkpoint files
- YOLO and VOC annotation formats
- Automatic anchor box generation
- Preconfigured to output training data to TensorBoard
- Fixed or cyclic learning rates 
- Command-line backend interface 
- Human-in-the-loop prediction and dataset expansion

##### Development Goals:

- [OBS Studio](https://github.com/obsproject/obs-studio) utility for USB camera arrays (*in progress*)
- Statistical report generation using [traces](https://github.com/datascopeanalytics/traces) (*in progress*)
- TensorFlow 2 native code (*separate development branch created*)
- All darknet layer types implemented
- YOLOv3 detection
- Take advantage of python 3.8's multiprocessing.shared_memory


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

### Open SLGR-Suite
From the source directory run:
```
./slgrSuite.py
```

## Controls
|  Hotkey  |                     Action                     |
|----------|------------------------------------------------|
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

## Image Annotation
### Verify Image

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when expanding a training dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

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


## How to contribute

Send a pull request

## License

### Free software:
- [GPLv3](https://github.com/rjdbcm/slgrSuite/blob/master/LICENSE)
- [MIT NOTICE](https://github.com/rjdbcm/slgrSuite/blob/master/NOTICE)

### Based in part on original code by: 
- Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg
- Mahmoud Aslan. Cyclic Learning Rate. Git code (2018). https://github.com/mhmoodlan/cyclic-learning-rate


Related
-------

1. [labelImg](https://github.com/tzutalin/labelImg) the original image annotation software SLGR-Suite is forked from
2. [darkflow](https://github.com/thtrieu/darkflow) the original basis of the machine learning backend
3. [cyclic-learning-rate](https://github.com/mhmoodlan/cyclic-learning-rate) the implementation of cyclic learning rates used
4. [traces](https://github.com/datascopeanalytics/traces) library for non-transformative unevenly-spaced timeseries analysis
5. [OBS Studio](https://github.com/obsproject/obs-studio) video recording software
6. [You Only Look Once:Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
7. [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
8. [A Framework for the Analysis of Unevenly Spaced Time Series Data](http://www.eckner.com/papers/unevenly_spaced_time_series_analysis.pdf)
9. [Unevenly-spaced data is actually pretty great](https://datascopeanalytics.com/blog/unevenly-spaced-time-series/) 
10. [Interactive machine learning: experimental evidence for the human in the algorithmic loop](https://link.springer.com/content/pdf/10.1007/s10489-018-1361-5.pdf)
11. [Why Momentum Really Works](https://distill.pub/2017/momentum/)
