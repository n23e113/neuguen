# -*- coding: utf-8 -*-
"""
decode batch example from tfrecord

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
from tensorflow.contrib import slim

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset')
DEFAULT_CONFIG = {
    'name': 'celeba',
    'training_size': 90000,
    'test_size': 900,
    'pattern_training_set': 'celeba_training*.tfrecord',
    'pattern_test_set': 'celeba_test*.tfrecord',
    'face_image_shape': (218, 178, 3),
    'num_of_classes': 2,
    'items_to_descriptions': {
        ''
    }
}
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])
DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)

def config_to_slim_dataset(config=None, dataset_dir=None):
#
#Args:
#    config: dataset config
#
#Returns:
#    slim.dataset.Dataset
#
    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR

    if not config:
        config = DEFAULT_CONFIG

    zero = tf.zeros([1], dtype=tf.int64)
    keys_to_features = {
        'image_face/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image_face/format':
        tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'label':
        tf.FixedLenFeature([1], tf.int64, default_value=zero),
    }

    items_to_handlers = {
        'image_face':
        slim.tfexample_decoder.Image(
            shape=config['face_image_shape'],
            image_key='image_face/encoded',
            format_key='image_face/format'),
        'label':
        slim.tfexample_decoder.Tensor('label'),

    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)

    file_pattern_training_set = os.path.join(
        dataset_dir, config['pattern_training_set'])

    file_pattern_test_set = os.path.join(
        dataset_dir, config['pattern_test_set'])

    training_set = slim.dataset.Dataset(
      data_sources=file_pattern_training_set,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config['training_size'],
      items_to_descriptions=config['items_to_descriptions'])

    test_set = slim.dataset.Dataset(
      data_sources=file_pattern_test_set,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config['test_size'],
      items_to_descriptions=config['items_to_descriptions'])

    return training_set, test_set

def slim_dataset_to_prefetch_queue(dataset, batch_size, shuffle=True):
#Args:
#    dataset: slim.dataset.Dataset
#    batch_size: batch size
#Returns:
#    slim.prefetch_queue.prefetch_queue contain face image batch tensor and emoji image batch tensor

    shuffle_config = DEFAULT_SHUFFLE_CONFIG

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=True,
        common_queue_capacity=16 * batch_size,
        common_queue_min=2 * batch_size)

    face_image, label = provider.get(['image_face', 'label'])

    if shuffle:
        face_image_batch, label_batch = tf.train.shuffle_batch(
            [face_image, label],
            batch_size=batch_size,
            num_threads=shuffle_config.num_batching_threads,
            capacity=shuffle_config.queue_capacity,
            min_after_dequeue=shuffle_config.min_after_dequeue)
    else:
        face_image_batch, label_batch = tf.train.batch(
            [face_image, label],
            batch_size=batch_size,
            num_threads=shuffle_config.num_batching_threads,
            capacity=shuffle_config.queue_capacity)

    # resize to 224 x 224 (h x w)
    face_image_batch = tf.cast(tf.image.resize_images(
        face_image_batch, [224, 224]), tf.uint8)

    return slim.prefetch_queue.prefetch_queue([face_image_batch, label_batch])
