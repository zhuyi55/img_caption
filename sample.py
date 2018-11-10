#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import tensorflow as tf
import vgg

from rnn_model import RNN_Model

from flags import parse_args
FLAGS, unparsed = parse_args()

slim = tf.contrib.slim


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

filename = r'./2513260012_03d33305cf.jpg'

is_training_placeholder = tf.placeholder(tf.bool)

with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

reverse_list = [reverse_dictionary[str(i)]
                for i in range(len(reverse_dictionary))]
                
image_data = tf.read_file(filename)
image_data = tf.image.decode_jpeg(image_data)
image_data = tf.image.resize_images(image_data, [224, 224])
image_data = tf.reshape(image_data, [1,224,224,3])

with slim.arg_scope(vgg.vgg_arg_scope()):
    cnn_logits, end_points = vgg.vgg_16(image_data,
                                    num_classes=FLAGS.dim_embedding,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='VALID')

logits_shape = tf.shape(cnn_logits)
data_x0 = tf.reshape(cnn_logits, [1, 1, FLAGS.dim_embedding])

#img_shape = tf.shape(image_tensor)
with tf.variable_scope('rnn'):
    rnn_m = RNN_Model(data_x0, is_training=False)
    rnn_m.convert()
    rnn_m.build()
                            
                            
log_folder = FLAGS.output_dir


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')
    

                            
    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')
        exit(0)

    state = sess.run(rnn_m.state_tensor)
    
    feed_dict_to_use = {is_training_placeholder: False,
                        rnn_m.state_tensor: state,
                        rnn_m.SampleX: [[0]],
                        rnn_m.keep_prob: 1.0}
    
    pred, state = sess.run( [rnn_m.predictions, rnn_m.outputs_state_tensor], feed_dict=feed_dict_to_use)

    #word_index = pred[0].argsort()[-1][int(np.random.rand()*2)-2]
    word_index = pred[0].argsort()[-1][-1]
    sentence = np.take(reverse_list, word_index)
    
    # generate sample
    for i in range(64):
        feed_dict = {is_training_placeholder: False,
                     rnn_m.state_tensor: state,
                     rnn_m.SampleX: [[word_index]],
                     rnn_m.keep_prob: 1.0}
    
        pred, state = sess.run(
            [rnn_m.predictions, rnn_m.outputs_state_tensor], feed_dict=feed_dict)
    
        #word_index = pred[0].argsort()[-1][int(-np.random.rand()*2)-2]
        word_index = pred[0].argsort()[-1][-1]
        word = np.take(reverse_list, word_index)
        sentence = sentence + ' ' + word
        
        if (word == '.'):
            break
            
    print (sentence)
    
    logging.debug('==============[{0}]=============='.format(filename))
    logging.debug(sentence)
