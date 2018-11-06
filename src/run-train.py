#!/usr/bin/env python

"""
Keras implementation of simplified DeepSpeech model.
Forked and adjusted from: https://github.com/robmsmt/KerasDeepSpeech
"""

import argparse
import os
import sys
from datetime import datetime
from os import makedirs
from os.path import join, abspath, isdir, exists
from shutil import rmtree

from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam

from core.batch_generator import CSVBatchGenerator
from core.models import *
from core.report_callback import ReportCallback
from util.ctc_util import get_tokens
from util.log_util import create_args_str
from util.rnn_util import load_keras_model, create_keras_session

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
parser.add_argument('--decoder', type=str, default='beamsearch,bestpath',
                    help='decoder to use (\'beamsearch\' or \'bestpath\') for validation. Default: None (both)')
parser.add_argument('--lm', type=str,
                    help='path to KenLM binary file to use for validation')
parser.add_argument('--lm_vocab', type=str, required=False,
                    help='(optional) path to text file containing vocabulary used to train KenLM model. The vocabulary '
                         'must be words separated by a single whitespace without newlines. A vocabulary is mandatory '
                         'if a LM is supplied with \'--lm_path\'. If \'--lm_path\' is set and  lm_vocab_path not, a '
                         'default vocabulary file with the same name as lm_path and the ending \'.vocab\' '
                         'will be searched. If this is not found, the script will exit.')
parser.add_argument('--language', type=str, choices=['en', 'de', 'fr', 'it', 'es'], default='en',
                    help='language to train on. '
                         'English will use 26 characters from the alphabet, German 29 (umlauts)')
parser.add_argument('--dropouts', action='store_true',
                    help='whether to use dropouts (default: False)')
parser.add_argument('--use_synth', action='store_true',
                    help='use synthesized training data if available (default: False)')
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd',
                    help='(optional) optimizer to use. Default=SGD')
parser.add_argument('--tensorboard', type=bool, default=True, help='True/False to use tensorboard')
parser.add_argument('--train_batches', type=int, default=0,
                    help='number of batches to use for training in each epoch. Use 0 for automatic')
parser.add_argument('--valid_batches', type=int, default=0,
                    help='number of batches to use for validation in each epoch. Use 0 for automatic')
parser.add_argument('--n_fc', type=int, default=512, help='fully connected size for model')
parser.add_argument('--n_recurrent', type=int, default=512, help='size of the RNN layer')
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
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(date_time):
    print(create_args_str(args))

    target_dir = setup(date_time)
    print()
    print(f'all output will be written to {target_dir}')
    print()

    print(f'creating {args.optimizer.upper()} optimizer for model')
    if args.optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    else:
        opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = create_model(target_dir, opt, args.dropouts, args.language)

    train_model(model, args.language, target_dir, args.minutes)


def setup(date_time):
    if not args.run_id:
        args.run_id = 'DS_' + date_time.strftime('%Y-%m-%d_%H-%M')

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

    args.decoder = args.decoder.split(',')

    return target_dir


def create_model(target_dir, opt, dropouts, language):
    tokens = get_tokens(language)
    n_labels = len(tokens) + 1  # +1 for blank token!
    print(f'using {n_labels} labels in output layer')

    if args.model_path:
        print(f'trying to load model from {target_dir}')
        if not isdir(args.model_path):
            print(f'ERROR: directory {target_dir} does not exist!', file=sys.stderr)
            exit(0)
        model = load_keras_model(target_dir, opt)
    else:
        if dropouts:
            print('Creating new model with dropouts')
            model = deep_speech_dropout(n_features=26, n_fc=args.n_fc, n_recurrent=args.n_recurrent, n_labels=n_labels)
        else:
            print('Creating new model without dropouts')
            model = deep_speech_lstm(n_features=26, n_fc=args.n_fc, n_recurrent=args.n_recurrent, n_labels=n_labels)
        model.compile(optimizer=opt, loss=ctc)

    model.summary()

    return model


def train_model(model, language, target_dir, num_minutes=None):
    create_keras_session(args.gpu)
    print("Creating data batch generators")
    data_train = CSVBatchGenerator(args.train_files, lang=language, sort=True, n_batches=args.train_batches,
                                   batch_size=args.batch_size, num_minutes=num_minutes, use_synth=args.use_synth)
    data_valid = CSVBatchGenerator(args.valid_files, lang=language, sort=False, n_batches=args.valid_batches,
                                   batch_size=args.batch_size)

    tensorboard_path = join(target_dir, 'tensorboard')
    if exists(tensorboard_path):
        rmtree(tensorboard_path)
    tb_cb = TensorBoard(log_dir=tensorboard_path, write_graph=False, write_images=True)

    report_cb = ReportCallback(data_valid, model, language, num_minutes=num_minutes, num_epochs=args.epochs,
                               target_dir=target_dir, lm_path=args.lm, vocab_path=args.lm_vocab)

    model.fit_generator(generator=data_train,
                        validation_data=data_valid,
                        steps_per_epoch=len(data_train),
                        validation_steps=len(data_valid),
                        epochs=args.epochs,
                        callbacks=[tb_cb, report_cb])

    K.clear_session()


if __name__ == '__main__':
    start_time = datetime.now()
    print(f'Training run started on : {start_time}')
    main(start_time)
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f'Training run finished on : {end_time}. Total time: {total_time}')
