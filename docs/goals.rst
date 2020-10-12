##########################
Darknet Layers Implemented
##########################

Part of this project converts darknet configurations to their equivalent
Tensorflow operations. One of the goals of this project is to bring the BEAGLES
darknet backend to parity with Darknet proper.

Progress:

   .. image:: https://progress-bar.dev/60

The following checklist tracks the progress toward that goal:

   | ☑ activation - Not handled as a layer, more of a decoration for other layers.

      | ☐ logistic
      | ☐ loggy
      | ☑ relu
      | ☑ elu
      | ☐ selu
      | ☐ gelu
      | ☐ relie
      | ☐ ramp
      | ☐ linear
      | ☐ tanh
      | ☐ psle
      | ☑ leaky
      | ☑ stair
      | ☑ hardtan
      | ☐ lhtan

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
   - Simplified and more portable checkpointing with the saved_model API
   - Improved code maintainability

Progress:

   .. image:: https://progress-bar.dev/50

