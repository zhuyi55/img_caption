#!/usr/bin/env python3
import logging

import cv2
from vgg import vgg_16

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(data)
    ##img_mask = image2label(img_label)
    ##encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(data.encode('utf8')),
        'image/encoded': _bytes_feature(encoded_data),
        'image/label': _bytes_feature(label.encode('utf8')),
        'image/format': _bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_record(output_filename, file_list, file_pars):

  writer = tf.python_io.TFRecordWriter(output_filename)

  idx = 0
  discard = 0
  used = 0
  for data, label in file_pars:
    if idx % 500 == 0:
      print ("dispose data index=%d" % (idx))
      
    idx += 1
    #  logging.info('On image %d of %d', idx, len(file_pars))
    #  print ("on image:%s with label:%s" % (data, label) )

    #if not os.path.exists(data):
    #  logging.warning('Could not find %s, ignoring example data.', xml_path)
    #  continue
    #
    #if not os.path.exists(label):
    #  logging.warning('Could not find %s, ignoring example label.', xml_path)
    #  continue
    if (data not in file_list):
        continue

    try:
      tf_example = dict_to_tf_example(data, label)
      if (tf_example != None):
        writer.write(tf_example.SerializeToString())
        used += 1
      else:
        discard += 1
    except ValueError:
      print ("value Error:%s", ValueError)
      logging.warning('Invalid example: %s, ignoring.', xml_path)

  print ("%d pics in record. discard=%d" % (used, discard))
  writer.close()

def read_token_files(root):
    token_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr8k.token.txt')
    
    with open(token_fname, 'r') as f:
        data = []
        label = []
        for each in f:
            line = each.split('\t')
            data.append('%s/data_Flickr8k/Flickr8k_Dataset/%s' % (root, line[0][:-2]))
            label.append(line[1][:-1])
    
    return zip(data, label)

def read_images_names(root, type='train'):
    txt_fname = ''
    if type == 'train':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.trainImages.txt')
    if type == 'test':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.testImages.txt')
    if type == 'val':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.devImages.txt')

    images = []
    with open(txt_fname, 'r') as f:
        for each in f:
            images.append('%s/data_Flickr8k/Flickr8k_Dataset/%s' % (root, each[:-1]))

    return images
    
    ##data = []
    ##label = []
    ##for fname in images:
    ##    data.append('%s/JPEGImages/%s.jpg' % (root, fname))
    ##    label.append('%s/SegmentationClass/%s.png' % (root, fname))
    ##return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_val.record')
    test_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_test.record')

    train_files = read_images_names(FLAGS.data_dir, 'train')
    val_files = read_images_names(FLAGS.data_dir, 'val')
    test_files = read_images_names(FLAGS.data_dir, 'test')

    token_files = read_token_files(FLAGS.data_dir)
    create_tf_record(train_output_path, train_files, token_files)

    token_files = read_token_files(FLAGS.data_dir)
    create_tf_record(val_output_path, val_files, token_files)

    token_files = read_token_files(FLAGS.data_dir)
    create_tf_record(test_output_path, test_files, token_files)


if __name__ == '__main__':
    tf.app.run()