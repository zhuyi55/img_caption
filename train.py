#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import logging

import numpy as np
import tensorflow as tf
import vgg

from dataset import inputs, read_and_decode

from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

slim = tf.contrib.slim

tf.reset_default_graph()
is_training_placeholder = tf.placeholder(tf.bool)
batch_size = FLAGS.batch_size

image_tensor_train, label_tensor_train = inputs(FLAGS.dataset_train, train=True, batch_size=batch_size, num_epochs=1e4)
image_tensor_val, label_tensor_val = inputs(FLAGS.dataset_val, train=False, num_epochs=1e4)

image_tensor, label_tensor = tf.cond(is_training_placeholder,
                                                           true_fn=lambda: (image_tensor_train, label_tensor_train),
                                                           false_fn=lambda: (image_tensor_val, label_tensor_val))

feed_dict_to_use = {is_training_placeholder: True}

log_folder = FLAGS.output_dir
#log_folder = os.path.join(FLAGS.output_dir, 'train')

vgg_checkpoint_path = FLAGS.checkpoint_path

# Creates a variable to hold the global_step.
global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)


# Define the model that we want to use -- specify to use only two classes at the last layer
with slim.arg_scope(vgg.vgg_arg_scope()):
    cnn_logits, end_points = vgg.vgg_16(image_tensor,
                                    num_classes=FLAGS.dim_embedding,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='VALID')

logits_shape = tf.shape(cnn_logits)

#img_shape = tf.shape(image_tensor)

##model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
##model.build(FLAGS.embedding_file)

with tf.variable_scope('embedding'):
    if FLAGS.embedding_file:
        # if embedding file provided, use it.
        embedding = np.load(FLAGS.embedding_file)
        embed = tf.constant(embedding, name='embedding')
    else:
        # if not, initialize an embedding and train it.
        print ('no embedding file.!')
        embed = tf.get_variable(
            'embedding', [FLAGS.num_words, FLAGS.dim_embedding])
        tf.summary.histogram('embed', embed)

    data = tf.nn.embedding_lookup(embed, label_tensor)

    data_x0 = tf.reshape(cnn_logits, [1, 1, FLAGS.dim_embedding])

    rnn_Y = label_tensor
    label_shape = tf.shape(label_tensor)
    
    concat_Data = tf.concat([data, data_x0], 1)
    _, rnn_X = tf.split(concat_Data, [1, -1], 1)

with tf.variable_scope('rnn'):
    basic_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.dim_embedding)
    drop_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, input_keep_prob=FLAGS.keep_prob, output_keep_prob=1.0)
    mult_cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell] * FLAGS.rnn_layers)
    
    state_tensor = mult_cell.zero_state(FLAGS.batch_size, tf.float32)
    seq_output, state_tensor = tf.nn.dynamic_rnn(mult_cell, rnn_X, initial_state=state_tensor)

    # flatten it
    seq_output_final = tf.reshape(seq_output, [-1, FLAGS.dim_embedding])
    
    with tf.variable_scope('softmax'):
        W_o = tf.get_variable('W_o', [FLAGS.dim_embedding, FLAGS.num_words], initializer=tf.random_normal_initializer(stddev=0.01))
        b_o = tf.get_variable('b_o', [FLAGS.num_words], initializer=tf.constant_initializer(0.0))
        
        rnn_logits = tf.reshape(tf.matmul(seq_output_final, W_o) + b_o, [-1, label_shape[1], FLAGS.num_words])
        
    tf.summary.histogram('rnn_logits', rnn_logits)
    
    predictions = tf.nn.softmax(rnn_logits, name='predictions')
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rnn_Y, logits = rnn_logits)
    mean, var = tf.nn.moments(rnn_logits, -1)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('logits_loss', loss)
    
    var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
    tf.summary.scalar('var_loss', var_loss)
    # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
    loss = loss + var_loss
    tf.summary.scalar('total_loss', loss)

with tf.variable_scope('adam_vars'):
    # gradient clip
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer = train_op.apply_gradients(
        zip(grads, tvars), global_step=global_step)

merged_summary_op = tf.summary.merge_all()

#lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
#cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                          labels=lbl_onehot)
#
#cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))


# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
##pred = tf.argmax(logits, axis=3)
##
##probabilities = tf.nn.softmax(logits)

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_loss)
##with tf.variable_scope("adam_vars"):
##    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#    gradients = optimizer.compute_gradients(loss=cross_entropy_loss)

 #   for grad_var_pair in gradients:
 #
 #       current_variable = grad_var_pair[1]
 #       current_gradient = grad_var_pair[0]
 #
 #       # Relace some characters from the original variable name
 #       # tensorboard doesn't accept ':' symbol
 #       gradient_name_to_save = current_variable.name.replace(":", "_")
 #
 #       # Let's get histogram of gradients for each layer and
 #       # visualize them later in tensorboard
 #       tf.summary.histogram(gradient_name_to_save, current_gradient)

    #train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
    ##train_step = tf.Variable(0)
    
# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
vgg_weights = slim.get_variables_to_restore(include=['vgg_16'], exclude=['vgg_16/fc8'])

# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
#vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

#rnn_weights = slim.get_variables_to_restore(include=['rnn'])

# Add summary op for the loss -- to be able to see it in
# tensorboard.
#tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

# Put all summary ops into one op. Produces string when
# you run it.
#merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

checkpoint_path = tf.train.latest_checkpoint(log_folder)
continue_train = False
if checkpoint_path:
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % log_folder)
    variables_to_restore = slim.get_model_variables()

    continue_train = True

else:
    # Create an OP that performs the initialization of
    # values of variables to the values from VGG.
    read_vgg_weights_func = slim.assign_from_checkpoint_fn(
        vgg_checkpoint_path,
        vgg_weights)

    # Initializer for new fc8 weights -- for two classes.
    #vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

    # Initializer for adam variables
    optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)


with sess:
    # Run the initializers.
    sess.run(init_op)
    sess.run(init_local_op)
    if continue_train:
        saver.restore(sess, checkpoint_path)

        logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))
    else:
        #sess.run(vgg_fc8_weights_initializer)
        sess.run(optimization_variables_initializer)

        read_vgg_weights_func(sess)
        logging.debug('value initialized...')

    # start data reader
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start = time.time()
    for i in range(FLAGS.max_steps):
        gs, _, state, l, summary_string, lista, listb = sess.run(
            [global_step, optimizer, state_tensor, loss, merged_summary_op, image_tensor, label_tensor], feed_dict=feed_dict_to_use)
        summary_string_writer.add_summary(summary_string, gs)

        print (lista)
        print (listb)
        if gs % 1 == 0:
            logging.debug('step [{0}] loss [{1}]'.format(gs, l))
        if gs % 5 == 0:
            save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=gs)
            logging.debug("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
    logging.debug("Model saved in file: %s" % save_path)

summary_string_writer.close()
