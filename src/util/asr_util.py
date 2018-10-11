import numpy as np
import pandas as pd
from keras import backend as K
from python_speech_features import mfcc

from util.lm_util import correction, ler_norm, wer, ler, wers, lers, lers_norm
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

    return pd.DataFrame({'ground truth': ground_truths,
                         'greedy': preds_greedy,
                         'beam': preds_beam,
                         'greedy+LM': [correction(pred_greedy, lm, lm_vocab) for pred_greedy in preds_greedy],
                         'beam+LM': [correction(pred_beam, lm, lm_vocab) for pred_beam in preds_beam]
                         })


def calculate_metrics(inferences):
    results = []
    for ix, (ground_truth, pred_greedy, pred_greedy_lm, pred_beam, pred_beam_lm) in inferences.iterrows():
        index = pd.MultiIndex.from_product([['greedy', 'beam'], ['lm_n', 'lm_y']],
                                           names=['decoding strategy', 'LM correction'])
        result = pd.DataFrame(index=index, columns=['ground truth', 'prediction', 'WER', 'LER', 'LER_raw'])
        result['ground truth'] = ground_truth
        result.loc['greedy', 'lm_n']['prediction'] = pred_greedy
        result.loc['greedy', 'lm_y']['prediction'] = pred_greedy_lm
        result.loc['beam', 'lm_n']['prediction'] = pred_beam
        result.loc['beam', 'lm_y']['prediction'] = pred_beam_lm

        result.loc['greedy', 'lm_n']['LER'] = ler_norm(ground_truth, pred_greedy)
        result.loc['greedy', 'lm_y']['LER'] = ler_norm(ground_truth, pred_greedy_lm)
        result.loc['beam', 'lm_n']['LER'] = ler_norm(ground_truth, pred_beam)
        result.loc['beam', 'lm_y']['LER'] = ler_norm(ground_truth, pred_beam_lm)

        result.loc['greedy', 'lm_n']['WER'] = wer(ground_truth, pred_greedy)
        result.loc['greedy', 'lm_y']['WER'] = wer(ground_truth, pred_greedy_lm)
        result.loc['beam', 'lm_n']['WER'] = wer(ground_truth, pred_beam)
        result.loc['beam', 'lm_y']['WER'] = wer(ground_truth, pred_beam_lm)

        result.loc['greedy', 'lm_n']['LER_raw'] = ler(ground_truth, pred_greedy)
        result.loc['greedy', 'lm_y']['LER_raw'] = ler(ground_truth, pred_greedy_lm)
        result.loc['beam', 'lm_n']['LER_raw'] = ler(ground_truth, pred_beam)
        result.loc['beam', 'lm_y']['LER_raw'] = ler(ground_truth, pred_beam_lm)

        results.append(result)

    return pd.concat(results)


def calculate_metrics_mean(inferences):
    index = pd.MultiIndex.from_product([['greedy', 'beam'], ['lm_n', 'lm_y']],
                                       names=['decoding strategy', 'LM correction'])

    df = pd.DataFrame(index=index, columns=['Ø WER', 'Ø LER', 'Ø LER (raw)'])
    df.loc['greedy', 'lm_n']['Ø WER'] = wers(inferences['ground truth'], inferences['greedy']).mean()
    df.loc['greedy', 'lm_y']['Ø WER'] = wers(inferences['ground truth'], inferences['greedy+LM']).mean()
    df.loc['beam', 'lm_n']['Ø WER'] = wers(inferences['ground truth'], inferences['beam']).mean()
    df.loc['beam', 'lm_y']['Ø WER'] = wers(inferences['ground truth'], inferences['beam+LM']).mean()

    df.loc['greedy', 'lm_n']['Ø LER'] = lers_norm(inferences['ground truth'], inferences['greedy']).mean()
    df.loc['greedy', 'lm_y']['Ø LER'] = lers_norm(inferences['ground truth'], inferences['greedy+LM']).mean()
    df.loc['beam', 'lm_n']['Ø LER'] = lers_norm(inferences['ground truth'], inferences['beam']).mean()
    df.loc['beam', 'lm_y']['Ø LER'] = lers_norm(inferences['ground truth'], inferences['beam+LM']).mean()

    df.loc['greedy', 'lm_n']['Ø LER (raw)'] = lers(inferences['ground truth'], inferences['greedy']).mean()
    df.loc['greedy', 'lm_y']['Ø LER (raw)'] = lers(inferences['ground truth'], inferences['greedy+LM']).mean()
    df.loc['beam', 'lm_n']['Ø LER (raw)'] = lers(inferences['ground truth'], inferences['beam']).mean()
    df.loc['beam', 'lm_y']['Ø LER (raw)'] = lers(inferences['ground truth'], inferences['beam+LM']).mean()

    return df
