import numpy as np
from keras import backend as K
from python_speech_features import mfcc

from util.rnn_util import decode


def infer_transcription(model, audio, rate):
    feature = mfcc(audio, samplerate=rate, numcep=26)
    input_length = np.array([feature.shape[0]])

    input_date = model.get_layer('the_input').input[0]
    ctc = model.get_layer('ctc')
    K.set_learning_phase(0)
    test_func = K.function([input_date, K.learning_phase()], [ctc])

    y_pred = test_func([[feature]])
    decoded_int = K.get_value(K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=False))
    decoded_str = [decode(int_seq) for int_seq in decoded_int]
    K.clear_session()

    return decoded_str[0]
