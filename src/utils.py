import resource
import types
from os import makedirs
from os.path import exists, join

import keras
import keras.backend as K
from keras.engine.saving import load_model
from keras.models import model_from_json
from pympler import muppy, summary, tracker
from pympler.web import start_in_background

from model import clipped_relu, selu, ctc


# these text/int characters are modified
# from the DS2 github.com/baidu-research/ba-dls-deepspeech


def save_trimmed_model(model, name):
    jsonfilename = str(name) + ".json"
    weightsfilename = str(name) + ".h5"

    # # serialize model to JSON
    with open(jsonfilename, "w") as json_file:
        json_file.write(model.to_json())

    # # serialize weights to HDF5
    model.save_weights(weightsfilename)

    return


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


def load_model_checkpoint(root_path, opt=None):
    # this is a terrible hack
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


memlist = []


class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        web_browser_debug = True
        print(x)

        if x > 40000:
            if web_browser_debug:
                if epoch == 0:
                    start_in_background()
                    tr = tracker.SummaryTracker()
                    tr.print_diff()
            else:
                global memlist
                all_objects = muppy.get_objects(include_frames=True)
                # print(len(all_objects))
                sum1 = summary.summarize(all_objects)
                memlist.append(sum1)
                summary.print_(sum1)
                if len(memlist) > 1:
                    # compare with last - prints the difference per epoch
                    diff = summary.get_diff(memlist[-2], memlist[-1])
                    summary.print_(diff)
                my_types = muppy.filter(all_objects, Type=types.ClassType)

                for t in my_types:
                    print(t)

    #########################################################
