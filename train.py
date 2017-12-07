#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
from PIL import Image
import numpy as np
import tensorflow as tf
import gen_dataset.data_provider as data_provider
import model.model_build as model_build

logging.basicConfig(level=logging.INFO)

A_DUMMY_CONFIG = {
    'name': 'celeba',
    'training_size': 40,
    'test_size': 40,
    'pattern_training_set': 'gen_dataset/test.conf.tfrecord',
    'pattern_test_set': 'gen_dataset/test.conf.tfrecord',
    'face_image_shape': (218, 178, 3),
    'num_of_classes': 2,
    'items_to_descriptions': {
        ''
    }}

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

def train():
    dataset, testset = data_provider.config_to_slim_dataset(
        config=TRAINING_CONFIG, dataset_dir="./")

    # training data
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, BATCH_SIZE)
    face_batch, label_batch = prefetch_queue.dequeue()
    face_batch = tf.cast(face_batch, tf.float32)

    tf.summary.image("face", face_batch[0:16], max_outputs=16)

    # testing data
    #prefetch_queue_test = data_provider.slim_dataset_to_prefetch_queue(testset, 256)
    #face_test_batch, label_test_batch = prefetch_queue_test.dequeue()
    #face_test_batch = tf.cast(face_test_batch, tf.float32)

    x = tf.placeholder(tf.uint8, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.int64, shape=(None, 1))

    logit, trainable, total_reg_losses = model_build.build_mobilenet_v1(x)
    tf.summary.scalar("regularization_loss", tf.reduce_sum(total_reg_losses))

    loss = model_build.build_loss(logit, y)
    tf.summary.scalar("cross entropy loss", loss)

    loss = loss + tf.reduce_sum(total_reg_losses)
    tf.summary.scalar("total loss", loss)

    global_step = tf.contrib.slim.create_global_step()
    partial_train_op, full_train_op = model_build.build_train_op(loss, trainable, global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(logit, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("batch accuracy", accuracy)
    confusion_matrix_op = tf.confusion_matrix(tf.squeeze(y), tf.argmax(logit, 1))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    neuguen_saver = tf.train.Saver(max_to_keep=10)
    merge_summary = tf.summary.merge_all()

    save_path = "neuguen_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session(config=session_config) as session:
        summary_writer = tf.summary.FileWriter(save_path, session.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())

        model_build.restore_pretrained_mobilenet(session)

        for j in xrange(100):
            confusion_matrix = np.array([[0., 0.], [0., 0.]])
            accuracy_avg = 0.0
            if j < 40:
                train_op = partial_train_op
            else:
                train_op = full_train_op
            for i in xrange(int(TRAINING_CONFIG["training_size"] / BATCH_SIZE)):
                faces, labels, step = session.run([face_batch, label_batch, global_step])
                if step % 100 == 99:
                    if step % 1000 == 999:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, loss_value, accuracy_value, confusion, _ = session.run(
                            [merge_summary, loss, accuracy, confusion_matrix_op, train_op],
                            feed_dict={x: faces, y: labels}, options=run_options,
                            run_metadata=run_metadata)
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_run_metadata(run_metadata, "step{0}".format(step))
                    else:
                        summary, loss_value, accuracy_value, confusion, _ = session.run(
                            [merge_summary, loss, accuracy, confusion_matrix_op, train_op],
                            feed_dict={x: faces, y: labels})
                        summary_writer.add_summary(summary, step)
                else:
                    loss_value, accuracy_value, confusion, _ = session.run(
                        [loss, accuracy, confusion_matrix_op, train_op],
                        feed_dict={x: faces, y: labels})
                confusion_matrix = confusion_matrix + confusion
                accuracy_avg = accuracy_avg + (accuracy_value - accuracy_avg) / (i + 1)
                sys.stdout.write("\r{0}--{1} training accuracy(ma):{2}    ".format(j, i, accuracy_avg))
                sys.stdout.flush()
            print("")
            print(confusion_matrix)

            neuguen_saver.save(session, os.path.join(save_path, "neuguen.ckpt"), global_step=global_step)

            #faces, labels = session.run([face_test_batch, label_test_batch])
            #accuracy_value, confusion = session.run(
            #    [accuracy, confusion_matrix_op], feed_dict={x: faces, y: labels})
            #print("{0} test accuracy:{1}    ".format(j, accuracy_value))
            #print(confusion_matrix)
            #print("")

        print("thread.join")
        coord.request_stop()
        coord.join(threads)

train()
