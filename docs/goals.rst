##########################
Darknet Layers Implemented
##########################

Part of this project converts darknet configurations to their equivalent
Tensorflow operations. One of the goals of this project is to bring the BEAGLES
darknet backend to parity with Darknet proper.

Progress:

   .. image:: https://progress-bar.dev/80

The following checklist tracks the progress toward that goal:
   | ☑ activation - Not handled as a layer, more of a decoration for other layers.

      | ☑ logistic
      | ☑ loggy
      | ☑ relu
      | ☑ elu
      | ☑ selu
      | ☑ gelu
      | ☑ relie
      | ☐ ramp
      | ☑ linear
      | ☑ tanh
      | ☑ psle
      | ☑ leaky
      | ☑ stair
      | ☑ hardtan
      | ☑ softplus
      | ☑ lhtan

   | ☑ avgpool
   | ☑ batchnorm - Not handled as a layer, more of a decoration for other layers.
   | ☑ connected
   | ☑ conv-lstm
   | ☑ convolutional
   | ☑ cost - Not handled as layer, uses :obj:`beagles.backend.framework.NeuralNet`
   | ☐ crnn
   | ☑ crop
   | ☐ deconvolutional
   | ☑ detection - Not handled as a layer, Uses :obj:`beagles.backend.framework.Yolo`
   | ☑ dropout
   | ☐ Gaussian-yolo
   | ☑ gru
   | ☑ local
   | ☑ lstm
   | ☑ maxpool
   | ☐ normalization
   | ☑ region - Not handled as a layer, Uses :obj:`beagles.backend.framework.YoloV2`
   | ☑ reorg
   | ☑ rnn
   | ☐ sam
   | ☐ scale-channels
   | ☑ shortcut
   | ☑ softmax
   | ☑ upsample
   | ☐ yolo - May need to use a combination of Framework and Layer API

#######################
Migrate to Tensorflow 2
#######################

Currently there is a mix of Tensorflow 1.x and Tensorflow 2 APIs but it is a
goal to remove all :obj:`tensorflow.compat.v1` symbols from the BEAGLES codebase.

There are several advantages to migrating:

   - Simplified summary API
   - Simplified and more portable checkpointing
   - Improved performance with :meth:`tensorflow.function` decorator
   - Improved code maintainability

Update 2020-Oct-22:
   Created a NetBuilder API. Currently assessing how much code can be deprecated
   by using the Keras API to manage weights and checkpoints.

Update 2020-Nov-03:
   Converted all code to TF 2.0. Keeping legacy code in case anyone still wants
   to toy with TF 1.x.
   3x the FPS performance for YOLOv2 detection.

Progress:

   .. image:: https://progress-bar.dev/95

###################################
Extend Darknet Configuration Format
###################################

Darknet is designed mainly with YOLO in mind and we would like to expand this.
For one thing, activation functions are canned and don't take arguments in Darknet.
Activations in BEAGLES are Keras layer objects that use the same BaseOp API as
all the other layers which allows arbitrary ops to be used as activations.

We have also added support for `Albumentations <https://albumentations.ai>`_
image augmentation pipelines for dataset expansion as the 'augment' keyword in
the [net] config file section. You can test the various augmentations `here <https://albumentations-demo.herokuapp.com>`_.

   | ☑ Arbitrary activation ops for darknet layers
   | ☑ Soft-NMS as a configuration file option using keyword 'soft_nms' for [detection] and [region] layers.
   | ☑ Image augmentations using [net] section keyword 'augment' and comma-separated `Albumentations <https://albumentations.ai>`_ transforms.

      | ☑ Tested :class:`Blur`
      | ☑ Tested :class:`ChannelDropout`
      | ☑ Tested :class:`ChannelShuffle`
      | ☑ Tested :class:`CLAHE`
      | ☑ Tested :class:`CoarseDropout` (not supported)
      | ☑ Tested :class:`ColorJitter`
      | ☑ Tested :class:`Crop` (not supported)
      | ☑ Tested :class:`CropNonEmptyMaskIfExists` (not supported)
      | ☑ Tested :class:`Cutout`
      | ☑ Tested :class:`Downscale`
      | ☑ Tested :class:`ElasticTransform` (not supported)
      | ☑ Tested :class:`Equalize`
      | ☑ Tested :class:`FancyPCA`
      | ☑ Tested :class:`Flip`
      | ☑ Tested :class:`FromFloat` (not supported)
      | ☑ Tested :class:`GaussianBlur`
      | ☑ Tested :class:`GaussNoise`
      | ☑ Tested :class:`GlassBlur`
      | ☑ Tested :class:`GridDistortion` (not supported)
      | ☑ Tested :class:`GridDropout` (not supported)
      | ☑ Tested :class:`HorizontalFlip`
      | ☑ Tested :class:`HueSaturationValue`
      | ☑ Tested :class:`ImageCompression`
      | ☑ Tested :class:`InvertImg`
      | ☑ Tested :class:`ISONoise`
      | ☑ Tested :class:`JpegCompression`
      | ☑ Tested :class:`Lambda` (not supported)
      | ☑ Tested :class:`LongestMaxSize` (not supported)
      | ☑ Tested :class:`MaskDropout` (not supported)
      | ☑ Tested :class:`MedianBlur`
      | ☑ Tested :class:`MotionBlur`
      | ☑ Tested :class:`MultiplicativeNoise`
      | ☑ Tested :class:`Normalize`
      | ☑ Tested :class:`OpticalDistortion`
      | ☑ Tested :class:`PadIfNeeded` (not supported)
      | ☑ Tested :class:`Posterize`
      | ☑ Tested :class:`RandomBrightness`
      | ☑ Tested :class:`RandomBrightnessContrast`
      | ☑ Tested :class:`RandomContrast`
      | ☑ Tested :class:`RandomCrop` (not supported)
      | ☑ Tested :class:`RandomCropNearBBox` (not supported)
      | ☑ Tested :class:`RandomFog`
      | ☑ Tested :class:`RandomGamma`
      | ☑ Tested :class:`RandomGridShuffle`
      | ☑ Tested :class:`RandomRain`
      | ☑ Tested :class:`RandomResizedCrop` (not supported)
      | ☑ Tested :class:`RandomRotate90`
      | ☑ Tested :class:`RandomScale`
      | ☑ Tested :class:`RandomShadow`
      | ☑ Tested :class:`RandomSizedBBoxSafeCrop` (not supported)
      | ☑ Tested :class:`RandomSizedCrop` (not supported)
      | ☑ Tested :class:`RandomSnow`
      | ☑ Tested :class:`RandomSunFlare`
      | ☑ Tested :class:`Resize` (not supported)
      | ☑ Tested :class:`RGBShift`
      | ☑ Tested :class:`Rotate`
      | ☑ Tested :class:`ShiftScaleRotate`
      | ☑ Tested :class:`SmallestMaxSize` (not supported)
      | ☑ Tested :class:`Solarize`
      | ☑ Tested :class:`ToFloat` (not supported)
      | ☑ Tested :class:`ToGray`
      | ☑ Tested :class:`ToSepia`
      | ☑ Tested :class:`Transpose`
      | ☑ Tested :class:`VerticalFlip`
