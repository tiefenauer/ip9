"""
Contains various helper functions to create/train a BRNN
"""
from genericpath import exists
from os import makedirs
from os.path import join

from keras import backend as K
from keras.engine.saving import model_from_json


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


def load_model(root_path, opt=None):
    """
    Load model from directory
    :param root_path: directory with model files
    :param opt: optimizer to use (optional if model is loaded from HDF5 file
    :return: the compiled model
    """
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})

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