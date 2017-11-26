# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from PIL import Image

import gen_dataset.data_provider as data_provider

TEST_CONFIG = {
    'name': 'test',
    'size': 40,
    'pattern_training_set': 'test.*.tfrecord',
    'pattern_test_set': 'test.*.tfrecord',
    'face_image_shape': (218, 178, 3),
    'num_of_classes': 2,
    'items_to_descriptions': {
        ''
    }
}

def test():
    dataset, _ = data_provider.config_to_slim_dataset(config=TEST_CONFIG, dataset_dir="./")
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 1)
    face_batch, label_batch = prefetch_queue.dequeue()

    init = tf.global_variables_initializer()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        session.run(init)
        for i in xrange(10):

            faces, labels = session.run([face_batch, label_batch])
            im = Image.fromarray(faces[0])
            im.save("{0}_{1}_face.jpg".format(labels[0], i))

        print("thread.join")
        coord.request_stop()
        coord.join(threads)

test()
