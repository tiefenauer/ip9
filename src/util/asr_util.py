import numpy as np
from keras import backend as K
from python_speech_features import mfcc

from util.lm_util import correction
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


def infer_batch(batch_inputs, decoder_greedy, decoder_beam, lm=None, lm_vocab=None):
    batch_input = batch_inputs['the_input']
    batch_input_lengths = batch_inputs['input_length']
    ground_truths = batch_inputs['source_str']

    preds_greedy = decoder_greedy.decode(batch_input, batch_input_lengths)
    preds_beam = decoder_beam.decode(batch_input, batch_input_lengths)

    preds_greedy_lm = [correction(pred_greedy, lm, lm_vocab) for pred_greedy in preds_greedy]
    preds_beam_lm = [correction(pred_beam, lm, lm_vocab) for pred_beam in preds_beam]

    return ground_truths, preds_greedy, preds_greedy_lm, preds_beam, preds_beam_lm
