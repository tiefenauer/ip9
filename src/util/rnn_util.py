"""
Contains various helper functions to create/train a BRNN
"""
from genericpath import exists
from os import makedirs
from os.path import join

import tensorflow as tf
from keras import backend as K
from keras.engine.saving import model_from_json, load_model
from keras.utils import get_custom_objects

from core.models import clipped_relu, ctc
import string
from abc import ABC

import numpy as np

# 29 target classes
# <space> = 0, a=1, b=2, ..., z=26, '=27, _ (padding token) = 28
SPACE_TOKEN = '<space>'
ALLOWED_CHARS = string.ascii_lowercase  # add umlauts here
CHAR_TOKENS = ' ' + ALLOWED_CHARS + '\''


def tokenize(text):
    """Splits a text into tokens.
    The text must only contain the lowercase characters a-z and digits. This must be ensured prior to calling this
    method for performance reasons. The tokens are the characters in the text. A special <space> token is added between
    the words. Since numbers are a special case (e.g. '1' and '3' are 'one' and 'three' if pronounced separately, but
    'thirteen' if the text is '13'), digits are mapped to the special '<unk>' token.
    """

    text = text.replace(' ', '  ')
    words = text.split(' ')

    tokens = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in words])
    return tokens


def encode(text):
    return [encode_token(token) for token in tokenize(text)]


def encode_token(token):
    return 0 if token == SPACE_TOKEN else CHAR_TOKENS.index(token)


def decode(tokens):
    return ''.join([decode_token(x) for x in tokens])


def decode_token(ind):
    return '' if ind in [-1, len(CHAR_TOKENS)] else CHAR_TOKENS[ind]



def save_model(model, target_dir):
    if not exists(target_dir):
        makedirs(target_dir)

    model_path = join(target_dir, 'model.h5')
    json_path = join(target_dir, 'arch.json')
    weights_path = join(target_dir, 'weights.h5')
    print(f'Saving model in {model_path}, weights in {weights_path} and architecture in {json_path}')

    model.save(model_path)
    model.save_weights(weights_path)
    with open(json_path, "w") as json_file:
        json_file.write(model.to_json())


def load_model_from_dir(root_path, opt=None):
    """
    Load model from directory
    :param root_path: directory with model files
    :param opt: optimizer to use (optional if model is loaded from HDF5 file
    :return: the compiled model
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"ctc": ctc})

    # prefer single file over architecture + weight
    model_h5 = join(root_path, 'model.h5')
    if exists(model_h5):
        print(f'loading model from {model_h5}')
        K.set_learning_phase(1)
        return load_model(model_h5)

    model_arch = join(root_path, "arch.json")
    if not exists(model_arch):
        raise ValueError(f'ERROR: No HDF5 model found at {root_path} and also no architecture found at {model_arch}!')

    model_weights = join(root_path, "weights.h5")
    if not exists(model_arch):
        raise ValueError(f'ERROR: architecture found in {model_arch}, but no weights in {model_weights}')

    if not opt:
        raise ValueError(f'ERROR: you must supply an optimizer when trying to load from architecture/weights !')

    with open(model_arch, 'r') as json_file:
        print(f'loading model architecture from {model_arch} and weights from {model_weights}')
        loaded_model_json = json_file.read()

        K.set_learning_phase(1)
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        loaded_model.compile(optimizer=opt, loss=ctc)
        return loaded_model


def create_keras_session(gpu, allow_growth=False, log_device_placement=False):
    if gpu is None:
        gpu = input('Enter GPU to use: ')
        print(f'GPU set to {gpu}')

    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.visible_device_list = gpu
    config.gpu_options.allow_growth = allow_growth
    session = tf.Session(config=config)
    K.set_session(session)