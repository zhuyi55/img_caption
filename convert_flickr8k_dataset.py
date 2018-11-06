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
import json

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '.', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '.', 'Path to directory to output TFRecords.')
flags.DEFINE_string('json_dictionary', './dictionary.json', 'The dictionary transfer words to index')
#flags.DEFINE_string('label_length', 32, 'Max label length for image label.')

FLAGS = flags.FLAGS

def dict_to_tf_example(data, sentense):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img = cv2.imread(data)
    ##img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    ##img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ##img = img / 255
    ##img = img.astype(np.float32)

    height, width = img.shape[0], img.shape[1]
    ##if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
    ##    # 保证最后随机裁剪的尺寸
    ##    return None

#    print (sentense.tostring())
#    print (type(sentense.tostring()))
    
    # Your code here, fill the dict
    feature_dict = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(data.encode('utf8')),
        'image/encoded': _bytes_feature(encoded_data),
        'image/label': _bytes_feature(sentense.tostring()),
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

  print(len(file_pars))
  for idx in range( len(file_pars) ):
    if idx % 10000 == 0:
      print ("dispose data index=%d" % (idx))

    data = file_pars[idx][0]
    label = file_pars[idx][1]
    #  logging.info('On image %d of %d', idx, len(file_pars))
    #  print ("on image:%s with label:%s" % (data, label) )

    if not os.path.exists(data):
      logging.warning('Could not find %s, ignoring example data.', data)
      continue

    if (data not in file_list):
        continue
    
    word_indexs = index_data(label)
    #word_indexs = label
    try:
      tf_example = dict_to_tf_example(data, word_indexs)
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

    return list(zip(data, label))

def read_images_names(root, type='train'):
    txt_fname = ''
    if type == 'train':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.trainImages.txt')
    if type == 'test':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.testImages.txt')
    if type == 'val':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.devImages.txt')
    if type == 'one':
        txt_fname = os.path.join(root, 'data_Flickr8k/', 'Flickr_8k.oneImages.txt')
        
    images = []
    with open(txt_fname, 'r') as f:
        for each in f:
            images.append('%s/data_Flickr8k/Flickr8k_Dataset/%s' % (root, each[:-1]))

    return images

def index_data(label):
    with open(FLAGS.json_dictionary, encoding='utf-8') as inf:
        dictionary = json.load(inf, encoding='utf-8')

        words = label.split()
        if (words[-1] != '.'):
            words.append('.')
            
        index = [dictionary['UNK']] * len(words)
        for i in range(len(words)):
            try:
                index[i] = dictionary[words[i]]
            except KeyError:
                index[i] = dictionary['UNK']

        return np.array(index)
    
    return None

def main(_):
    logging.info('Prepare dataset file names')
    
    token_files = read_token_files(FLAGS.data_dir)
    
    ## generate one tfrecord for debug
    test_files = read_images_names(FLAGS.data_dir, 'one')
    test_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_train_one.record')
    create_tf_record(test_output_path, test_files, token_files)

    ## generate train tfrecord
    train_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_train.record')
    train_files = read_images_names(FLAGS.data_dir, 'train')
    create_tf_record(train_output_path, train_files, token_files)

    ## generate validate tfrecord
    val_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_val.record')
    val_files = read_images_names(FLAGS.data_dir, 'val')
    create_tf_record(val_output_path, val_files, token_files)

    ## generate test tfrecord
    test_files = read_images_names(FLAGS.data_dir, 'test')
    test_output_path = os.path.join(FLAGS.output_dir, 'flickr8k_test.record')
    create_tf_record(test_output_path, test_files, token_files)


if __name__ == '__main__':
    tf.app.run()
