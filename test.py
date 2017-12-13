#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import argparse
import random
random.seed(1)
from PIL import Image
import numpy as np
import tensorflow as tf
import gen_dataset.data_provider as data_provider
import model.model_build as model_build

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", help="specify checkpoint file")
args = parser.parse_args()

TRAINING_CONFIG = {
    'name': 'celeba',
    'training_size': 198599,
    'test_size': 4000,
    'pattern_training_set': 'gen_dataset/celeba.train.tfrecord',
    'pattern_test_set': 'gen_dataset/celeba.test.tfrecord',
    'face_image_shape': (218, 178, 3),
    'num_of_classes': 2,
    'items_to_descriptions': {''}
}

BATCH_SIZE = 64
DEFAULT_MODEL_PATH="neuguen_model"

def test():
    dataset, testset = data_provider.config_to_slim_dataset(
        config=TRAINING_CONFIG, dataset_dir="./")

    # testing data
    prefetch_queue_test =\
        data_provider.slim_dataset_to_prefetch_queue(testset,
            BATCH_SIZE, shuffle=False)
    face_test_batch, label_test_batch = prefetch_queue_test.dequeue()
    face_test_batch = tf.cast(face_test_batch, tf.float32)

    x = tf.placeholder(tf.uint8, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.int64, shape=(None, 1))

    logit, _, _, _ = model_build.build_mobilenet_v1_debug(x, mobilenet_training=False, neuguen_training=False)

    global_step = tf.train.create_global_step()
    increment_global_step_op = tf.assign(global_step, global_step+1)

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(logit, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_matrix_op = tf.confusion_matrix(tf.squeeze(y), tf.argmax(logit, 1))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    neuguen_saver = tf.train.Saver()
    save_path = "neuguen_test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())

        if args.model_checkpoint:
            neuguen_saver.restore(session, args.model_checkpoint)
        else:
            neuguen_saver.restore(session, tf.train.latest_checkpoint(DEFAULT_MODEL_PATH))

        for j in xrange(1):
            confusion_matrix = np.array([[0., 0.], [0., 0.]])
            accuracy_avg = 0.0
            for i in xrange(int(TRAINING_CONFIG["test_size"] / BATCH_SIZE)):
                faces, labels, step = session.run([face_test_batch, label_test_batch, increment_global_step_op])
                accuracy_value, confusion, logit_value = session.run(
                    [accuracy, confusion_matrix_op, logit],
                    feed_dict={x: faces, y: labels})
                confusion_matrix = confusion_matrix + confusion
                accuracy_avg = accuracy_avg + (accuracy_value - accuracy_avg) / (i + 1)
                sys.stdout.write("\r{0}--{1} training accuracy(ma):{2}    ".format(j, i, accuracy_avg))
                sys.stdout.flush()
            print("")
            print(confusion_matrix)

        print("thread.join")
        coord.request_stop()
        coord.join(threads)

test()
