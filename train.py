#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from PIL import Image
import numpy as np
import tensorflow as tf
import gen_dataset.data_provider as data_provider
import model.model_build as model_build

logging.basicConfig(level=logging.INFO)

TEST_CONFIG = {
    'name': 'celeba',
    'size': 40,
    'pattern_training_set': 'gen_dataset/*.tfrecord',
    'pattern_test_set': 'gen_dataset/*.tfrecord',
    'face_image_shape': (218, 178, 3),
    'num_of_classes': 2,
    'items_to_descriptions': {
        ''
    }}

def train():
    dataset, testset = data_provider.config_to_slim_dataset(
        config=TEST_CONFIG, dataset_dir="./")
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 32)

    face_batch, label_batch = prefetch_queue.dequeue()
    face_batch = tf.cast(face_batch, tf.float32)

    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.int64, shape=(None, 1))

    logit, trainable, losses = model_build.build_model_fc8(x)
    loss = model_build.build_loss(logit, y)
    loss = loss #+ tf.reduce_sum(losses)

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(logit, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_op = model_build.build_train_op(loss, trainable)

    slim = tf.contrib.slim
    global_step = slim.create_global_step()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()

    vgg_saver = tf.train.Saver()

    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(init)

        """
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=["vgg_16"])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            "model/vgg_16.ckpt", variables_to_restore)
        init_fn(session)
        """

        for i in xrange(500):
            faces, labels = session.run([face_batch, label_batch])

            loss_value, accuracy_value, _ = session.run(
                [loss, accuracy, train_op], feed_dict={x: faces, y: labels})
            logging.info("loss {0} - {1} - {2}".format(i, loss_value, accuracy_value))

        print("thread.join")
        coord.request_stop()
        coord.join(threads)

train()
