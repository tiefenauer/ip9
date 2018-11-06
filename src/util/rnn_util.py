"""
Contains various helper functions to create/train a BRNN
"""
from genericpath import exists
from os import makedirs
from os.path import join

import tensorflow as tf
from deepspeech import Model
from keras import backend as K
from keras.engine.saving import model_from_json, load_model
from keras.utils import get_custom_objects
from tensorflow.python.client import device_lib

from core.models import clipped_relu, ctc


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


def load_ds_model(model_path, alphabet_path, lm_path=None, trie_path=None, n_features=26, n_context=9, beam_width=500,
                  lm_weight=1.75, valid_word_count_weight=1.00):
    print(f'loading DeepSpeech model from {model_path}, using alphabet at {alphabet_path}, '
          f'LM at {lm_path} and trie at {trie_path}')
    ds = Model(model_path, n_features, n_context, alphabet_path, beam_width)
    if lm_path and trie_path:
        ds.enableDecoderWithLM(alphabet_path, lm_path, trie_path, lm_weight, valid_word_count_weight)
    return ds


def load_keras_model(root_path, opt=None):
    """
    Load model from directory
    :param root_path: directory with model files
    :param opt: optimizer to use (optional if model is loaded from HDF5 file
    :return: the compiled model
    """
    print(f'loading Keras model from {root_path}')
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


def create_tf_session(gpu, allow_growth=False, log_device_placement=False):
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.visible_device_list = gpu
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(config=config)


def query_gpu(gpu):
    all_gpus = get_available_gpus()
    if all_gpus and gpu is None:
        gpu = input('Enter GPU to use (leave blank for all GPUs): ')
        if not gpu:
            gpu = ','.join(all_gpus)
        print(f'GPU set to {gpu}')
    return gpu


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
