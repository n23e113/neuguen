#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import model.vgg as vgg
import model.mobilenet_v1 as mobilenet_v1

def mobilenet_preprocess(inputs):
    if inputs.dtype != tf.float32:
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
        print("mobilenet preprocess convert image dtype")
    inputs = tf.subtract(inputs, 0.5)
    inputs = tf.multiply(inputs, 2.0)
    return inputs

def build_mobilenet_v1_logit(face_batch):

    face_batch = mobilenet_preprocess(face_batch)

    with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        logits, end_points = mobilenet_v1.mobilenet_v1(face_batch, num_classes=1001)

    slim = tf.contrib.slim
    var_scope_name = "neuguen"
    with tf.variable_scope(var_scope_name):
        logits = tf.nn.relu6(logits)
        logits = tf.contrib.layers.fully_connected(
                logits, 2, activation_fn=None, scope="fc1")

    return logits,\
           tf.trainable_variables(var_scope_name),\
           tf.losses.get_regularization_losses(var_scope_name)

def build_mobilenet_v1(face_batch, is_training=True):

    face_batch = mobilenet_preprocess(face_batch)

    with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)):
        _, end_points = mobilenet_v1.mobilenet_v1(face_batch, num_classes=1001)

    Conv2d_13_pointwise = end_points["Conv2d_13_pointwise"] # 7 x 7 x 1024
    Conv2d_13_pointwise = tf.image.resize_images(
        Conv2d_13_pointwise, [28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    Conv2d_10_pointwise = end_points["Conv2d_10_pointwise"] # 14 x 14 x 512
    Conv2d_10_pointwise = tf.image.resize_images(
            Conv2d_10_pointwise, [28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    Conv2d_7_pointwise = end_points["Conv2d_7_pointwise"] # 14 x 14 x 512
    Conv2d_7_pointwise = tf.image.resize_images(
            Conv2d_7_pointwise, [28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    Conv2d_4_pointwise = end_points["Conv2d_4_pointwise"] # 28 x 28 x 256

    feature_map = tf.concat(
        [Conv2d_13_pointwise,
         Conv2d_10_pointwise,
         Conv2d_7_pointwise,
         Conv2d_4_pointwise,
        ], 3)

    slim = tf.contrib.slim
    var_scope_name = "neuguen"
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                feature_map = slim.conv2d(feature_map, 64, 3, 1,
                                          normalizer_fn=slim.batch_norm)
                flatten = tf.contrib.layers.flatten(feature_map)
                logits = tf.contrib.layers.fully_connected(
                    flatten, 2, activation_fn=None, scope="fc1")

    return logits,\
           tf.trainable_variables(var_scope_name),\
           tf.losses.get_regularization_losses(var_scope_name)

def restore_pretrained_mobilenet(session):
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=["MobilenetV1"])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        "model/mobilenet_v1_1.0_224.ckpt", variables_to_restore, ignore_missing_vars=True)
    init_fn(session)

def restore_pretrained_vgg(session):
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=["vgg_16"])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        "model/vgg_16.ckpt", variables_to_restore, ignore_missing_vars=True)
    init_fn(session)

def vgg_process(face_batch):
    # vgg preprocess
    R_MEAN = 123.68
    G_MEAN = 116.78
    B_MEAN = 103.94
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=face_batch * 1.0)
    processed = tf.concat(values=[r - R_MEAN, g - G_MEAN, b - B_MEAN],
                    axis=3)
    return processed

def build_model_vgg16(face_batch):
    """

    :param face_batch:
    :return:
        logit
        trainable variable
        regularization losses
    """

    face_batch = vgg_process(face_batch)

    # use pretrained vgg to extract image feature
    # note 1, to restore vgg from standard pretrain checkpoint, number of classes must be set to 1000 (because of imagenet classes)
    # note 2, we do not train vgg's weights, so is_training(drop out) set to False
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        _, endpoints = vgg.vgg_16(processed, is_training=False)

    """
    we use:
        vgg.pool2 56x56x128
        vgg.conv3_3 56x56x256
        vgg.conv4_3 28x28x512 up pooling 56x56x512
        vgg.conv5_3 14x14x512 up pooling 56x56x512
    and concat
        56x56x1408
    then train a small convnet on these feature maps
        filter 3 x 3 x 512 output 56 x 56 x 512
        filter 3 x 3 x 128 output 56 x 56 x 128
        filter 3 x 3 x 16 output 56 x 56 x 16
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
        [pool2,
         conv3_3,
         conv4_3_up_pooling,
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
            feature_map = slim.conv2d(feature_map, 512, 3, 1)
            feature_map = slim.conv2d(feature_map, 128, 3, 1)
            feature_map = slim.conv2d(feature_map, 64, 3, 1)
            flatten = tf.contrib.layers.flatten(feature_map)
            logit = tf.contrib.layers.fully_connected(
                flatten, 2, activation_fn=None, scope="fc1")

    return logit,\
           tf.trainable_variables(var_scope_name),\
           tf.losses.get_regularization_losses(var_scope_name)

def build_model_vgg16_fc8(face_batch):
    """

    :param face_batch:
    :return:
        logit
        trainable variable
        regularization losses
    """

    face_batch = vgg_process(face_batch)

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        fc8, endpoints = vgg.vgg_16(bgr, is_training=False, dropout_keep_prob=1.0)

    slim = tf.contrib.slim
    var_scope_name = "neuguen"
    with tf.variable_scope(var_scope_name):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME'):
            fc8 = tf.nn.relu(fc8)
            logit = tf.contrib.layers.fully_connected(
                fc8, 2, activation_fn=None, scope="fc1")

    return logit,\
           tf.trainable_variables(var_scope_name),\
           tf.losses.get_regularization_losses(var_scope_name)
    #flatten = tf.contrib.layers.flatten(fc8)
    #return tf.contrib.slim.fully_connected(flatten, 2, activation_fn=None), None, None
    #return fc8, None, None

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

def build_simplest_model(face_batch):
    slim = tf.contrib.slim
    scope_name = "simplest"

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        to_next_layer = slim.conv2d(face_batch, 64, 3, 1)
        to_next_layer = slim.conv2d(to_next_layer, 64, 3, 1)
        out = slim.max_pool2d(to_next_layer, [2, 2], scope='pool1')
        flatten = tf.contrib.layers.flatten(out)
        return slim.fully_connected(flatten, 2, activation_fn=None), None, None

def build_vgg_part(inputs):

    inputs = vgg_process(inputs)

    slim = tf.contrib.slim
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        with tf.variable_scope("vgg_16", 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                              outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                flatten = tf.contrib.layers.flatten(net)
                return slim.fully_connected(flatten, 2, activation_fn=None), None, None


def build_simple_conv_model(face_batch):
    slim = tf.contrib.slim
    scope_name = "another"
    reuse = False
    image = face_batch
    image_channel = 3
    coding_len = 2
    convolution_repeat_times = 5
    filter_count = 16
    """
    Args:
        face image -> face image auto encoder
        scope_name: variable scope name
        reuse:
        image: batch image
        image_channel: image channel count, default is 3
        coding_len: auto encoder hidden coding length
        convolution_repeat_times: conv(rev conv) layer count
        filter_count: number of conv filter
    Returns:
        out: image
        coding: auto encoder hidden coding
        variables: trainable variable
    """

    # todo ricker, add batch norm
    # todo ricker, add sigmoid after coding
    print("filter_count:{0} coding_len:{1}".format(filter_count, coding_len))
    print("image shape {0}".format(image.shape))

    with tf.variable_scope(scope_name, reuse=reuse) as vs:
        # encoder
        to_next_layer = slim.conv2d(image, filter_count, 3, 1, activation_fn=tf.nn.elu)

        for idx in range(convolution_repeat_times):
            channel_num = filter_count * (idx + 1)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < convolution_repeat_times - 1:
                to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 2, activation_fn=tf.nn.elu)
                print("to_next_layer shape {0}".format(to_next_layer.shape))
        print("to_next_layer shape {0}".format(to_next_layer.shape))

        # hidden coding
        to_next_layer = tf.contrib.layers.flatten(to_next_layer)
        print("to_next_layer shape {0}".format(to_next_layer.shape))
        coding = to_next_layer = slim.fully_connected(to_next_layer, coding_len, activation_fn=None)
    return coding,\
           tf.trainable_variables(scope_name),\
           tf.losses.get_regularization_losses(scope_name)

def build_train_op(loss, trainable):
    """

    :param loss:
    :param trainable:
    :return:
    """
    optimizer = tf.train.AdamOptimizer()
    #optimizer = tf.train.GradientDescentOptimizer(0.0001)
    #return optimizer.minimize(loss)
    grad_var = optimizer.compute_gradients(loss, var_list=trainable)
    #grad_var = optimizer.compute_gradients(loss)
    return optimizer.apply_gradients(grad_var)
