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
from PIL import Image
import numpy as np
import tensorflow as tf
import gen_dataset.data_provider as data_provider
import model.model_build as model_build

random.seed(1)
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", help="specify checkpoint file", required=True)
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

    logit, _, _ = model_build.build_mobilenet_v1(x)

    global_step = tf.contrib.slim.create_global_step()

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(logit, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("batch accuracy", accuracy)
    confusion_matrix_op = tf.confusion_matrix(tf.squeeze(y), tf.argmax(logit, 1))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    neuguen_saver = tf.train.Saver()

    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        neuguen_saver.restore(session, args.model_checkpoint)

        for j in xrange(1):
            confusion_matrix = np.array([[0., 0.], [0., 0.]])
            accuracy_avg = 0.0
            for i in xrange(int(TRAINING_CONFIG["test_size"] / BATCH_SIZE)):
                faces, labels, step = session.run([face_test_batch, label_test_batch, global_step])
                accuracy_value, confusion = session.run(
                    [accuracy, confusion_matrix_op],
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
