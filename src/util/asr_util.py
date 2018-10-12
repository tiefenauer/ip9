import itertools

import numpy as np
import pandas as pd
from keras import backend as K
from python_speech_features import mfcc
from tqdm import tqdm

from util.lm_util import ler_norm, wer, ler, correction
from util.rnn_util import decode

decoding_strategies = ['greedy', 'beam']
lm_uses = ['lm_n', 'lm_y']
metrics = ['WER', 'LER', 'LER_raw']


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


def infer_batches(batch_generator, decoder_greedy, decoder_beam, lm, lm_vocab):
    batch_generator.cur_index = 0  # reset index
    inferences = []
    for _ in tqdm(range(len(batch_generator))):
        batch_inputs, _ = next(batch_generator)
        batch_inferences = infer_batch(batch_inputs, decoder_greedy, decoder_beam, lm, lm_vocab)
        inferences.append(batch_inferences)

    df_inferences = pd.concat(inferences, sort=False)
    df_inferences.index.name = 'ground truth'
    return df_inferences


def infer_batch(batch_inputs, decoder_greedy, decoder_beam, lm=None, lm_vocab=None):
    batch_input = batch_inputs['the_input']
    batch_input_lengths = batch_inputs['input_length']
    ground_truths = batch_inputs['source_str']

    preds_greedy = decoder_greedy.decode(batch_input, batch_input_lengths)
    preds_beam = decoder_beam.decode(batch_input, batch_input_lengths)

    preds_greedy_lm = [correction(pred_greedy, lm, lm_vocab) for pred_greedy in preds_greedy]
    preds_beam_lm = [correction(pred_beam, lm, lm_vocab) for pred_beam in preds_beam]

    columns = pd.MultiIndex.from_product([decoding_strategies, lm_uses, ['prediction'] + metrics],
                                         names=['decoding strategy', 'LM correction', 'predictions'])

    df_batch_results = pd.DataFrame(index=ground_truths, columns=columns)
    for ground_truth, pred_greedy, pred_greedy_lm, pred_beam, pred_beam_lm in zip(ground_truths,
                                                                                  preds_greedy, preds_greedy_lm,
                                                                                  preds_beam, preds_beam_lm):
        df_batch_results.loc[ground_truth]['greedy', 'lm_n', 'prediction'] = pred_greedy
        df_batch_results.loc[ground_truth]['greedy', 'lm_n', 'WER'] = wer(ground_truth, pred_greedy)
        df_batch_results.loc[ground_truth]['greedy', 'lm_n', 'LER'] = ler_norm(ground_truth, pred_greedy)
        df_batch_results.loc[ground_truth]['greedy', 'lm_n', 'LER_raw'] = ler(ground_truth, pred_greedy)

        df_batch_results.loc[ground_truth]['greedy', 'lm_y', 'prediction'] = pred_greedy_lm
        df_batch_results.loc[ground_truth]['greedy', 'lm_y', 'WER'] = wer(ground_truth, pred_greedy_lm)
        df_batch_results.loc[ground_truth]['greedy', 'lm_y', 'LER'] = ler_norm(ground_truth, pred_greedy_lm)
        df_batch_results.loc[ground_truth]['greedy', 'lm_y', 'LER_raw'] = ler(ground_truth, pred_greedy_lm)

        df_batch_results.loc[ground_truth]['beam', 'lm_n', 'prediction'] = pred_beam
        df_batch_results.loc[ground_truth]['beam', 'lm_n', 'WER'] = wer(ground_truth, pred_beam)
        df_batch_results.loc[ground_truth]['beam', 'lm_n', 'LER'] = ler_norm(ground_truth, pred_beam)
        df_batch_results.loc[ground_truth]['beam', 'lm_n', 'LER_raw'] = ler(ground_truth, pred_beam)

        df_batch_results.loc[ground_truth]['beam', 'lm_y', 'prediction'] = pred_beam_lm
        df_batch_results.loc[ground_truth]['beam', 'lm_y', 'WER'] = wer(ground_truth, pred_beam_lm)
        df_batch_results.loc[ground_truth]['beam', 'lm_y', 'LER'] = ler_norm(ground_truth, pred_beam_lm)
        df_batch_results.loc[ground_truth]['beam', 'lm_y', 'LER_raw'] = ler(ground_truth, pred_beam_lm)

    return df_batch_results


def calculate_metrics_mean(df_inferences):
    index = pd.MultiIndex.from_product([decoding_strategies, lm_uses], names=['decoding strategy', 'LM correction'])
    df = pd.DataFrame(index=index, columns=metrics)

    for decoding_strategy, lm_used, metric in itertools.product(decoding_strategies, lm_uses, metrics):
        df.loc[decoding_strategy, lm_used][metric] = df_inferences[decoding_strategy, lm_used, metric].mean()

    df.columns = ['Ø WER', 'Ø LER', 'Ø LER (raw)']

    return df
