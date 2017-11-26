# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
import argparse
import os
import io
import sys

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--input_config", help="specify input config file", required=True)
"""
    input config file format
    label image_dir
    0 ./grinning_face
    1 ./neutral_face
"""
args = parser.parse_args()

g_face_image_height = 218
g_face_image_width = 178




def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def gen_sample(label, image_path):
    """

    :param label:
    :param image_path:
    :return:
    """
    face_file_content = open(image_path, 'r').read()

    sample = tf.train.Example(features=tf.train.Features(
        feature={
        'image_face/format': bytes_feature("jpg"),
        'image_face/encoded': bytes_feature(face_file_content),
        'label': int64_feature(label),
    }))
    return sample

def gen_dataset(input_config_filename):
    """

    :param input_config_filename:
    :return:
    """
    sample_source = []
    file_count = 0
    for filelineno, line in enumerate(io.open(input_config_filename, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        # label - sample_path
        data = line.split(" ")
        for f in os.listdir(data[1]):
            s = os.path.join(data[1], f)
            if os.path.isfile(s):
                sample_source.append((int(data[0]), s))
                file_count += 1
                if file_count % 1000 == 0:
                    sys.stdout.write("*")
                    sys.stdout.flush()
    print("")
    random.shuffle(sample_source)

    writer = tf.python_io.TFRecordWriter(os.path.basename(input_config_filename) + ".tfrecord")
    for s in sample_source:
        sample = gen_sample(s[0], s[1])
        writer.write(sample.SerializeToString())
    writer.close()

gen_dataset(args.input_config)
