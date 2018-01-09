Object Recognition
------------------

This is the second part **Object Recognition** in Advanced Topics in Image Analysis, which is mainly about plant identification task. The dataset is from [LifeCLEF](http://www.imageclef.org/lifeclef/2015/plant).

Introduction for all code
=========================

- `logs/` the log file about model training, you can run it by `tensorboard --logdir='./logs/'`
- `preprocess/` includes the files which do the segmentation of LeafScan.
- `generate_data.py` parse `Xml` file and gernerate the traing, validate, test set with only leaf scan pictures.
- `handle_orig_images.py` pre-process all  original images.
- `model.py` Building the model, mainly implemented by [Keras](https://keras.io/).

The CNN model framework
=======================

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 64)      1792      
_________________________________________________________________
activation_1 (Activation)    (None, 224, 224, 64)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856     
_________________________________________________________________
activation_2 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584    
_________________________________________________________________
activation_3 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168    
_________________________________________________________________
activation_4 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080    
_________________________________________________________________
activation_5 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 28, 28, 512)       1180160   
_________________________________________________________________
activation_6 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 28, 28, 512)       2359808   
_________________________________________________________________
activation_7 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 14, 14, 512)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 512)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 100352)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               51380736  
_________________________________________________________________
activation_8 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 87)                44631     
_________________________________________________________________
activation_9 (Activation)    (None, 87)                0         
=================================================================
Total params: 56,073,815
Trainable params: 56,073,815
Non-trainable params: 0
_________________________________________________________________

```
