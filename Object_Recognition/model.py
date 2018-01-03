from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import cv2

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, AveragePooling2D
)
from keras import optimizers
from keras.initializers import RandomNormal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization

batch_size    = 100
epochs        = 100
iterations    = 100
num_classes   = 87
dropout       = 0.25
weight_decay  = 0.0001
log_filepath  = './logs'


def scheduler(epoch):
    if epoch <= 30:
        return 0.01
    if epoch <= 60:
        return 0.005
    return 0.001


def build_model():
    # build the network
    model = Sequential()

    model.add(
        Conv2D(
            64,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal",
            input_shape=x_train.shape[1:]
        )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding = 'same'))

    model.add(Dropout(dropout))

    model.add(
        Conv2D(
            128,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding = 'same'))

    model.add(Dropout(dropout))

    model.add(
        Conv2D(
            256,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding = 'same'))

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # load the pickle file
    train_dict = pickle.load(open('./train.pickle', 'rb'))
    test_dict = pickle.load(open('./test.pickle', 'rb'))
    label_dict = pickle.load(open('./label_dict.pickle', 'rb'))

    all_training_ims = train_dict.keys()
    all_testing_ims = test_dict.keys()

    # load data
    x_train = []
    y_train = []

    # Using all the data for training.
    for im_id in all_training_ims[:100]:
        image = cv2.imread('./process_train/' + im_id + '.jpg')
        try:
            im = cv2.resize(image, (100, 100))
            x_train.append(im)
            y_train.append(label_dict[train_dict[im_id]])
        except:
            continue
    print(len(x_train))
    x_test = []
    y_test = []

    for im_id in all_testing_ims:
        image = cv2.imread('./process_test/' + im_id + '.jpg')
        try:
            im = cv2.resize(image, (100, 100))
            x_test.append(im)
            y_test.append(label_dict[test_dict[im_id]])
        except:
            continue

    print('{} samples will be trained'.format(len(y_train)))

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True, # randomly flip images
    )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    print(x_train.shape)
    datagen.fit(x_train)

    # start training
    model.fit_generator(
        datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size
        ),
        steps_per_epoch=iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=(x_test, y_test),
    )
    model.save('model.h5')