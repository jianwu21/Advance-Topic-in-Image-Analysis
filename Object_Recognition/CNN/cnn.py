from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name="conv2d")


def max_pool(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="pooled")


def cnn_model(feature, model):
    '''
    This is convolutional layers model
    '''
    # RGB image
    input_layers = tf.reshape(feature, [-1, 500, 500, 3])

    xs = tf.placeholder(tf.float32, [None, 500, 500, 3])
    ys = tf.placeholder(tf.float32, [None, 87])
    keep_prob = tf.placeholder(tf.float32)

    # conv_1 layer
    with tf.name_scope('conv-layer-1'):
        W_conv1 = weight_variable([5, 5, 3, 16]) # outsize=32 :  convolutions units
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # 100 * 100 * 32
        h_pooled_1 = max_pool(h_conv1) # 50 * 50 * 32

    # conv_2 layer
    with tf.name_scope('conv-layer-2'):
        W_conv2 = weight_variable([5,5,16,8]) # outsize=64
        b_conv2 = bias_variable([8])
        h_conv2 = tf.nn.relu(conv2d(h_pooled_1, W_conv2) + b_conv2) # 25 * 25 *64
        h_pooled_2 = max_pool(h_conv2) # 25 * 25 * 64

    # cnv_3 layer
    with tf.name_scope('conv-layer-3'):
        W_conv3 = weight_variable([25, 25, 8, 8]) # outsize=64
        b_conv3 = bias_variable([8])
        h_conv3 = tf.nn.relu(conv2d(h_pooled_2, W_conv3) + b_conv3) # 25 * 25 *64
        h_pooled_3 = max_pool(h_conv3) # 25 * 25 * 64

    # func1 layer
    with tf.name_scope('nn-layer-1'):
        W_fun1 = weight_variable([25*25*8, 1024])
        b_fun1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pooled_3, [-1, 25*25*8])
        h_fun2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun1) + b_fun1)
        h_fun2_drop = tf.nn.dropout(h_fun2, keep_prob)


    # func2 layer
    with tf.name_scope('nn-layer-2'):
        W_fun2 = weight_variable([1024, 87])
        b_fun2 = bias_variable([87])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())


if __name__ == '__main__':
    tf.app.run()
