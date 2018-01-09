## Assignment 2

This folder includes all code file which is related with assignment2.

- `logs/` the log file about model training, you can run it by `tensorboard --logdir='./logs/'`
- `preprocess/` includes the files which do the segmentation of LeafScan.
- `generate_data.py` parse `Xml` file and gernerate the traing, validate, test set with only leaf scan pictures.
- `handle_orig_images.py` pre-process all  original images.
- `model.py` Building the model, mainly implemented by [Keras](https://keras.io/).
