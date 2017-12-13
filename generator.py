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
parser.add_argument("--input_pic", help="specify input picture path, if not specified, use random")
parser.add_argument("--cls", help="specify picture class to generate, default smile", default="smile")
parser.add_argument("--steps", help="specify generation step count, default 100", default=100)
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

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def constraint_loss(x):
    # constraint x at range [-1.0, 1.0], otherwise output linear loss
    return tf.reduce_sum(tf.nn.relu(-(x + 1.0))) + tf.reduce_sum(tf.nn.relu(x - 1.0))

def generate():

    if args.input_pic:
        init = read_image_file(args.input_pic)
    else:
        init = random_init()

    with tf.variable_scope("neuguen_generator"):
        # h x w x rgb [0-255]
        x = tf.get_variable("picture", dtype=tf.float32, initializer=init)
        tf.summary.histogram("x", x)
    if args.cls == "smile":
        y = tf.constant([[1]], dtype=tf.int64, shape=(1, 1))
        print("smile")
    else:
        y = tf.constant([[0]], dtype=tf.int64, shape=(1, 1))
        print("not smile")

    logit, _, _, _ = model_build.build_mobilenet_v1_debug(x,
        mobilenet_training=False, neuguen_training=False, preprocess=False)

    loss = model_build.build_loss(logit, y)
    constraint = constraint_loss(x)
    tf.summary.histogram("constraint_loss", constraint)
    loss = loss + constraint

    global_step = tf.train.create_global_step()

    train_op = model_build.build_generator_train_op(loss, [x], global_step)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    neuguen_saver = tf.train.Saver()
    save_path = args.model_checkpoint

    merge_summary = tf.summary.merge_all()
    save_path_generator = "neuguen_generator"
    if not os.path.exists(save_path_generator):
        os.makedirs(save_path_generator)

    with tf.Session(config=session_config) as session:

        session.run(tf.global_variables_initializer())

        model_build.restore_last_checkpoint(session, save_path)

        summary_writer = tf.summary.FileWriter(save_path_generator, session.graph)

        for j in xrange(args.steps):
            loss_value, _, step_value, logit_value, summary = session.run([loss, train_op, global_step, logit, merge_summary])
            summary_writer.add_summary(summary, step_value)
            sys.stdout.write("\r{0}--{1}    ".format(step_value, loss_value))
            sys.stdout.flush()
            #print(np.argmax(logit_value[0]))
            #print(logit_value)
        print("")
        print(logit_value[0])
        print(stable_softmax(logit_value[0]))
        generated_image = session.run([x])
        generated_image = (np.asarray(generated_image[0][0]) + 1.0) / 2.0 * 255
        generated_image = generated_image.astype(np.uint8)
        #print(generated_image.shape)
        #print(generated_image.dtype)
        result = Image.fromarray(generated_image)
        result = result.resize((178, 218), Image.ANTIALIAS)
        result.save('out.jpg')

generate()
