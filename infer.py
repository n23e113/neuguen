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
import model.model_build as model_build

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", help="specify checkpoint file", required=True)
parser.add_argument("--input_pic", help="specify input picture path", required=True)
args = parser.parse_args()

def read_image_file(image_file_path):
    image = Image.open(image_file_path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = ((np.asarray(image, dtype=np.float32) / 255.0) - 0.5) * 2.0

    image = np.expand_dims(image, axis=0)
    init = tf.constant(image)
    return init

def random_init():
    random_array = np.random.rand(1, 224, 224, 3)
    random_array = (random_array.astype(np.float32) - 0.5) * 2.0
    return tf.constant(random_array)

def infer():

    x = read_image_file(args.input_pic)
    logit, _, _, _ = model_build.build_mobilenet_v1_debug(x,
        mobilenet_training=False, neuguen_training=False, preprocess=False)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    neuguen_saver = tf.train.Saver()
    save_path = args.model_checkpoint

    with tf.Session(config=session_config) as session:

        session.run(tf.global_variables_initializer())

        model_build.restore_last_checkpoint(session, save_path)
        logit_value = session.run([logit])
        print(logit_value)
        print(np.argmax(logit_value[0]))

infer()
