#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from flags import parse_args
FLAGS, unparsed = parse_args()

class RNN_Model():
    def __init__(self, X0=None, label_tensor=None, is_training=True):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = FLAGS.batch_size
        self.num_words = FLAGS.num_words
        self.dim_embedding = FLAGS.dim_embedding
        self.rnn_layers = FLAGS.rnn_layers
        self.learning_rate = FLAGS.learning_rate
        self.label_tensor = label_tensor
        self.X0 = X0
        self.is_training = is_training

    def convert(self):
        with tf.variable_scope('embedding'):
            if FLAGS.embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(FLAGS.embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            if self.label_tensor != None:
                data = tf.nn.embedding_lookup(embed, self.label_tensor)
                if self.X0 != None:
                    concat_Data = tf.concat([self.X0, data], 1)
                    _, self.rnn_X = tf.split(concat_Data, [1, -1], 1)
                    self.Y = self.label_tensor
                    self.Y_shape = tf.shape(self.Y)
                    self.num_steps = self.Y_shape[1]
                    self.X0 = None
                else:
                    self.rnn_X = data
                    self.num_steps = 1
            else:
                self.rnn_X = self.X0
                self.num_steps = 1
        
    def build(self):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.SampleX = tf.placeholder(
            tf.int32, shape=[1, None], name='sample_input')
            
        if self.is_training == False:
            self.label_tensor = self.SampleX
            self.convert()
        ##self.Y = tf.placeholder(
        ##    tf.int32, shape=[self.batch_size, None], name='rnn_label')
            
        #self.X = tf.get_variable('rnn_input', [self.batch_size, 1, self.dim_embedding], initializer=tf.constant_initializer(0.0))
        #self.Y = tf.get_variable('rnn_label', [self.batch_size, 1, self.dim_embedding], initializer=tf.constant_initializer(0.0))
        #self.cnn_X = tf.get_variable('cnn_input', [self.batch_size, 1, self.dim_embedding], initializer=tf.constant_initializer(0.0))
        #self.cnn_X = tf.Variable(dtype=tf.float32, shape=[self.batch_size, 1, self.dim_embedding], name='cnn_input')
        
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            
        with tf.variable_scope('rnn'):
            basic_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding)
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, input_keep_prob=self.keep_prob, output_keep_prob=1.0)
            mult_cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell] * self.rnn_layers)
            
            self.state_tensor = mult_cell.zero_state(self.batch_size, tf.float32)
            seq_output, self.outputs_state_tensor = tf.nn.dynamic_rnn(mult_cell, self.rnn_X, initial_state=self.state_tensor)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            W_o = tf.get_variable('W_o', [self.dim_embedding, self.num_words], initializer=tf.random_normal_initializer(stddev=0.01))
            b_o = tf.get_variable('b_o', [self.num_words], initializer=tf.constant_initializer(0.0))
            
            rnn_logits = tf.reshape(tf.matmul(seq_output_final, W_o) + b_o, 
                                [-1, self.num_steps, self.num_words])
            
        tf.summary.histogram('logits', rnn_logits)

        self.predictions = tf.nn.softmax(rnn_logits, name='predictions')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits = rnn_logits)
        mean, var = tf.nn.moments(rnn_logits, -1)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
