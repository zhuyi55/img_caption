#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import time

import tensorflow as tf
import os

#from utils import (_B_MEAN, _G_MEAN, _R_MEAN, _mean_image_subtraction)


# 图片信息
#NUM_CLASSES = 128

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/label': tf.FixedLenFeature([], tf.string)
    })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image = tf.reshape(image, [1,224,224,3])
    
    label = tf.decode_raw(features['image/label'], tf.int32)
    label = tf.reshape(label, [1, -1])

    return image, label


def inputs(data_set, train=True, batch_size=1, num_epochs=1):
    assert os.path.exists(data_set), '[{0}] not exist!!!'.format(data_set)
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([data_set], num_epochs=num_epochs)

    image, label = read_and_decode(filename_queue)

    return image, label
