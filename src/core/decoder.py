import itertools

import numpy as np
from keras import backend as K

from util.ctc_util import decode


class Decoder(object):
    def __init__(self, model, decode_strategy):
        if decode_strategy not in ['old', 'bestpath', 'beamsearch']:
            raise ValueError(f'ERROR: invalid value \'{decode_strategy}\' for decode_strategy')

        self.ctc_input = model.get_layer('ctc').input[0]
        self.input_data = model.get_layer('the_input').input
        self.test_func = K.function([self.input_data, K.learning_phase()], [self.ctc_input])
        self.decode_strategy = decode_strategy

    def decode(self, batch_input, batch_input_lenghts):
        y_pred = self.test_func([batch_input])[0]
        if self.decode_strategy == 'old':
            predictions = decode_batch_old(self.test_func, batch_input)
        else:
            predictions = decode_batch_keras(y_pred, batch_input_lenghts, self.decode_strategy == 'bestpath')
        return predictions


def decode_batch_keras(y_pred, input_length, greedy=True):
    # https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
    decoded_int = K.get_value(K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=greedy)[0][0])
    decoded_str = [decode(int_seq) for int_seq in decoded_int]
    return decoded_str


def decode_batch_old(test_func, word_batch):
    results = []
    y_pred = test_func([word_batch])[0]

    for out in y_pred:
        best = list(np.argmax(out, axis=1))
        merged = [k for k, g in itertools.groupby(best)]
        results.append(decode(merged))

    return results
