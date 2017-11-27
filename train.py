#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import gen_dataset.data_provider as data_provider
import model.model_build as model_build

def train():
    dataset = data_provider.config_to_slim_dataset()
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 64)

    face_batch, label_batch = prefetch_queue.dequeue()
    face_batch = tf.image.convert_image_dtype(face_batch, dtype=tf.float32)

    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.int64, shape=(None, 1))

    logit, trainable, losses = model_build.build_model(x)
    loss = model_build.build_loss(logit, y)
    loss = loss + tf.reduce_sum(losses)

    train_op = model_build.build_train_op(loss, trainable)

    slim = tf.contrib.slim
    global_step = slim.create_global_step()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()

    vgg_saver = tf.train.Saver()

    with tf.Session(config=session_config) as session:
        session.run(init)
        vgg_saver.restore(session, "vgg.ckpt")

        for i in xrange(100):
            faces, labels = session.run([face_batch, label_batch])
            loss, _ = session.run([loss, train_op], feed_dict={x: faces, y: labels})
            logging.info("{0}-{1}".format(i, loss))