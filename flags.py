#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz


tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./rnn_log',
                        help='path to save log and checkpoint.')

    parser.add_argument('--text', type=str, default='QuanSongCi.txt',
                        help='path to QuanSongCi.txt')

#    parser.add_argument('--num_steps', type=int, default=64,
#                        help='number of time steps of one sample.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size to use.')

    parser.add_argument('--num_words', type=int, default=3108,
                        help='numer of words to use.')

    parser.add_argument('--dim_embedding', type=int, default=128,
                        help='embedding dim in embedding.npy file to use.')

    parser.add_argument('--rnn_layers', type=int, default=3,
                        help='rnn layers to use.')

    parser.add_argument('--keep_prob', type=int, default=1,
                        help='keep_prob for drop-out to use.')

    parser.add_argument('--dictionary', type=str, default='dictionary.json',
                        help='path to dictionary.json.')

    parser.add_argument('--reverse_dictionary', type=str, default='reverse_dictionary.json',
                        help='path to reverse_dictionary.json.')

    parser.add_argument('--embedding_file', type=str, default='./embedding.npy',
                        help='path to embedding file.')
                        
#    parser.add_argument('--learning_rate', type=float, default=0.001,
#                        help='learning rate')

    parser.add_argument('--checkpoint_path', type=str, default='./data/vgg_16.ckpt')
    
#    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset_train', type=str, default='./flickr8k_train_one.record')
    
    parser.add_argument('--dataset_val', type=str, default='flickr8k_val.record')
    
    parser.add_argument('--dataset_test', type=str, default='flickr8k_test.record')
    
#    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=40000)
    
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    for x in dir(FLAGS):
        print(getattr(FLAGS, x))
