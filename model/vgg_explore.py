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

for k, v in endpoints.iteritems():
    print(k, v)