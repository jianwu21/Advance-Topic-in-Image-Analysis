from __future__ import absolute_import, division, print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# The application will be added here
def cnn_model(feature, model):
    '''
    This is convolutional layers model
    '''
    # RGB image
    input_layers = tf.reshape(feature, [-1, *, *, 3])

    # Covolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
    )

    # pooling layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2,
    )

    # Covolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5]
        padding='same',
        activation=tf.nn.relu,
    )

    # pooling layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
    )

    # Building the suitable cnn structure



if __name__ == '__main__':
    tf.app.run()
