#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.vgg as vgg

inputs = tf.placeholder(tf.float32, (None, 224, 224, 3), name='inputs')
r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs * 255.0)
VGG_MEAN = [103.939, 116.779, 123.68]

bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
    fc8, endpoints = vgg.vgg_16(bgr, is_training=False)

def build_model(face_batch, label_batch):
    """

    :param face_batch:
    :param label_batch:
    :return:
    """

    # vgg preprocess
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs * 1.0)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]],
                    axis=3)

    # use pretrained vgg to extract image feature
    # note 1, to restore vgg from standard pretrain checkpoint, number of classes must be set to 1000 (because of fc8's weights and biases dim)
    # note 2, we do not train vgg's weights, so is_training(drop out) set to False
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        fc8, endpoints = vgg.vgg_16(bgr, is_training=False)

    """
    we use:
        vgg.pool2 56x56x128
        vgg.conv3_3 56x56x256
        vgg.conv4_3 28x28x512 up pooling 56x56x512
        vgg.conv5_3 14x14x512 up pooling 56x56x512
    and concat
        56x56x1408
    then train a small convnet on these feature maps
        filter 3 x 3 x number_of_class output 56 x 56 x number_of_class
        reduce_mean([1, 2])
    """
