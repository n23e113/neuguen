#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import model.vgg as vgg

def build_model(face_batch):
    """

    :param face_batch:
    :return:
        logit
        trainable variable
        regularization losses
    """

    # vgg preprocess
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=face_batch * 1.0)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]],
                    axis=3)

    # use pretrained vgg to extract image feature
    # note 1, to restore vgg from standard pretrain checkpoint, number of classes must be set to 1000 (because of fc8's weights and biases dim)
    # note 2, we do not train vgg's weights, so is_training(drop out) set to False
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        _, endpoints = vgg.vgg_16(bgr, is_training=False)

    """
    we use:
        vgg.pool2 56x56x128
        vgg.conv3_3 56x56x256
        vgg.conv4_3 28x28x512 up pooling 56x56x512
        vgg.conv5_3 14x14x512 up pooling 56x56x512
    and concat
        56x56x1408
    then train a small convnet on these feature maps
        filter 3 x 3 x 2 output 56 x 56 x 2
        flatten fc output 2 classes
    """

    pool2 = endpoints["vgg_16/pool2"]
    conv3_3 = endpoints["vgg_16/conv3/conv3_3"]

    conv4_3 = endpoints["vgg_16/conv4/conv4_3"]
    conv4_3_up_pooling = tf.image.resize_images(
        conv4_3, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv5_3 = endpoints["vgg_16/conv5/conv5_3"]
    conv5_3_up_pooling = tf.image.resize_images(
        conv5_3, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    feature_map = tf.concat(
        [#pool2,
         #conv3_3,
         #conv4_3_up_pooling,
         conv5_3_up_pooling
        ], 3)

    slim = tf.contrib.slim
    var_scope_name = "neuguen"
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME'):
            #feature_map = slim.repeat(
            #    feature_map, 1, slim.conv2d, 8, [3, 3], scope='conv1')
            flatten = tf.contrib.layers.flatten(conv5_3)
            logit = tf.contrib.layers.fully_connected(
                flatten, 2, activation_fn=tf.nn.relu, scope="fc1")

    return logit,\
           tf.trainable_variables(var_scope_name),\
           tf.losses.get_regularization_losses(var_scope_name)

def build_model_fc8(face_batch):
    """

    :param face_batch:
    :return:
        logit
        trainable variable
        regularization losses
    """

    # vgg preprocess
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=face_batch * 1.0)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]],
                    axis=3)

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        fc8, endpoints = vgg.vgg_16(bgr, is_training=True, num_classes=2)


    return fc8, tf.no_op, tf.no_op

def build_loss(preds, labels):
    """

    :param preds: model predict logit
    :param labels: sparse label
    :return: loss tensor
    """
    labels = tf.squeeze(labels)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=preds))

def build_train_op(loss, trainable):
    """

    :param loss:
    :param trainable:
    :return:
    """
    optimizer = tf.train.AdamOptimizer()
    #grad_var = optimizer.compute_gradients(loss, var_list=trainable)
    grad_var = optimizer.compute_gradients(loss)
    return optimizer.apply_gradients(grad_var)
