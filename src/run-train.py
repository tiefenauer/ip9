#!/usr/bin/env python

"""
Keras implementation of simplified DeepSpeech model.
Forked and adjusted from: https://github.com/robmsmt/KerasDeepSpeech
"""

import argparse
import datetime
import os
import sys
from os import makedirs
from os.path import join, abspath, isdir

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

from generator import CSVBatchGenerator
from model import *
from report import ReportCallback
from util.log_util import create_args_str
from utils import load_model_checkpoint, MemoryCallback

#######################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Prevent pool_allocator message
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True, help='root directory for all output')
parser.add_argument('--run_id', type=str, default='',
                    help='id of run, used to set checkpoint save name. Default uses timestamp')
parser.add_argument('--train_files', type=str, default='',
                    help='list of all train files, seperated by a comma if multiple')
parser.add_argument('--valid_files', type=str, default='',
                    help='list of all validation files, seperate by a comma if multiple')
parser.add_argument('--decoder', type=str, default='beamsearch',
                    help='decoder to use (\'beamsearch\', \'bestpath\'). Default: beamsearch')
parser.add_argument('--lm', type=str,
                    help='path to KenLM binary file to use for validation')
parser.add_argument('--lm_vocab', type=str,
                    help='path to text file containing vocabulary used to train KenLM model. The vocabulary must'
                         'be words separated by a single whitespace without newlines')
parser.add_argument('--tensorboard', type=bool, default=True, help='True/False to use tensorboard')
parser.add_argument('--memcheck', type=bool, default=False, help='print out memory details for each epoch')
parser.add_argument('--train_batches', type=int, default=0,
                    help='number of batches to use for training in each epoch. Use 0 for automatic')
parser.add_argument('--valid_batches', type=int, default=0,
                    help='number of batches to use for validation in each epoch. Use 0 for automatic')
parser.add_argument('--fc_size', type=int, default=512, help='fully connected size for model')
parser.add_argument('--rnn_size', type=int, default=512, help='size of the RNN layer')
parser.add_argument('--model_path', type=str, default='',
                    help="""If value set, load the checkpoint in a folder minus name minus the extension (weights 
                       assumed as same name diff ext) e.g. --model_path ./checkpoints/ TRIMMED_ds_ctc_model/""")
parser.add_argument('--learning_rate', type=float, default=0.01, help='the learning rate used by the optimiser')
parser.add_argument('--sort_samples', type=bool, default=True,
                    help='sort utterances by their length in the first epoch')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--minutes', type=int, default=None,
                    help='Number of minutes of training data to use. Default: None (=all)')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size used to train the model')
parser.add_argument('--gpu', type=str, nargs='?', default='2', help='(optional) GPU(s) to use for training. Default: 2')
args = parser.parse_args()


def main():
    print(create_args_str(args))

    target_dir = setup()
    print()
    print(f'all output will be written to {target_dir}')
    print()

    opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = create_model(target_dir, opt)

    train_model(model, target_dir, args.minutes)


def setup():
    if not args.run_id:
        args.run_id = 'DS_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    target_dir = join(args.target_dir, args.run_id)
    if not isdir(target_dir):
        makedirs(target_dir)

    # log_file_path = join(output_dir, 'train.log')
    # redirect_to_file(log_file_path)

    # detect user here too
    if args.train_files == "" and args.valid_files == "":
        # if paths to file not specified, assume testing
        test_path = join('data', 'ldc93s1')
        args.train_files = abspath(join(test_path, "ldc93s1.csv"))
        args.valid_files = abspath(join(test_path, "ldc93s1.csv"))

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.visible_device_list = args.gpu
    # config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    return target_dir


def create_model(target_dir, opt):
    if args.model_path:
        print(f'trying to load model from {target_dir}')
        if not isdir(args.model_path):
            print(f'ERROR: directory {target_dir} does not exist!', file=sys.stderr)
            exit(0)
        model = load_model_checkpoint(target_dir, opt)
    else:
        print('Creating new model')
        # model = deep_speech_dropout(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        model = ds1(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        model.compile(optimizer=opt, loss=ctc)

    model.summary()
    return model


def train_model(model, target_dir, num_minutes=None):
    print("Creating data batch generators")
    data_train = CSVBatchGenerator(args.train_files, sort=False, n_batches=args.train_batches,
                                   batch_size=args.batch_size, num_minutes=num_minutes)
    data_valid = CSVBatchGenerator(args.valid_files, sort=True, n_batches=args.valid_batches,
                                   batch_size=args.batch_size)

    cb_list = []
    if args.memcheck:
        cb_list.append(MemoryCallback())

    if args.tensorboard:
        tb_cb = TensorBoard(log_dir=join(target_dir, 'tensorboard'), write_graph=False, write_images=True)
        cb_list.append(tb_cb)

    report_cb = ReportCallback(data_valid, model, num_minutes=num_minutes, num_epochs=args.epochs,
                               target_dir=target_dir, decode_strategy=args.decoder, lm_path=args.lm, vocab_path=args.lm_vocab)
    cb_list.append(report_cb)

    model.fit_generator(generator=data_train,
                        validation_data=data_valid,
                        steps_per_epoch=len(data_train),
                        validation_steps=len(data_valid),
                        epochs=args.epochs, callbacks=cb_list, workers=1)

    K.clear_session()


if __name__ == '__main__':
    main()
