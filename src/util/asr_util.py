import numpy as np
import pandas as pd
from keras import backend as K
from python_speech_features import mfcc
from tqdm import tqdm

from util.lm_util import correction, ler_norm, wer, ler
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


def infer_batches(batch_generator, decoder_greedy, decoder_beam, lm, lm_vocab):
    batch_generator.cur_index = 0  # reset index
    inferences = []
    for _ in tqdm(range(len(batch_generator))):
        batch_inputs, _ = next(batch_generator)
        batch_inferences = infer_batch(batch_inputs, decoder_greedy, decoder_beam, lm, lm_vocab)
        inferences.append(batch_inferences)

    df_inferences = pd.concat(inferences, sort=False)
    df_metrics = calculate_metrics(df_inferences)
    return df_inferences, df_metrics


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


def calculate_metrics(df_inferences):
    results = []
    for ix, (ground_truth, pred_greedy, pred_greedy_lm, pred_beam, pred_beam_lm) in df_inferences.iterrows():
        index = pd.MultiIndex.from_product([['greedy', 'beam'], ['lm_n', 'lm_y']],
                                           names=['decoding strategy', 'LM correction'])
        df_result = pd.DataFrame(index=index, columns=['ground truth', 'prediction', 'WER', 'LER', 'LER_raw'])
        df_result['ground truth'] = ground_truth
        df_result.loc['greedy', 'lm_n']['prediction'] = pred_greedy
        df_result.loc['greedy', 'lm_y']['prediction'] = pred_greedy_lm
        df_result.loc['beam', 'lm_n']['prediction'] = pred_beam
        df_result.loc['beam', 'lm_y']['prediction'] = pred_beam_lm

        df_result.loc['greedy', 'lm_n']['LER'] = ler_norm(ground_truth, pred_greedy)
        df_result.loc['greedy', 'lm_y']['LER'] = ler_norm(ground_truth, pred_greedy_lm)
        df_result.loc['beam', 'lm_n']['LER'] = ler_norm(ground_truth, pred_beam)
        df_result.loc['beam', 'lm_y']['LER'] = ler_norm(ground_truth, pred_beam_lm)

        df_result.loc['greedy', 'lm_n']['WER'] = wer(ground_truth, pred_greedy)
        df_result.loc['greedy', 'lm_y']['WER'] = wer(ground_truth, pred_greedy_lm)
        df_result.loc['beam', 'lm_n']['WER'] = wer(ground_truth, pred_beam)
        df_result.loc['beam', 'lm_y']['WER'] = wer(ground_truth, pred_beam_lm)

        df_result.loc['greedy', 'lm_n']['LER_raw'] = ler(ground_truth, pred_greedy)
        df_result.loc['greedy', 'lm_y']['LER_raw'] = ler(ground_truth, pred_greedy_lm)
        df_result.loc['beam', 'lm_n']['LER_raw'] = ler(ground_truth, pred_beam)
        df_result.loc['beam', 'lm_y']['LER_raw'] = ler(ground_truth, pred_beam_lm)

        results.append(df_result)

    return pd.concat(results)


def calculate_metrics_mean(df_metrics):
    index = pd.MultiIndex.from_product([['greedy', 'beam'], ['lm_n', 'lm_y']],
                                       names=['decoding strategy', 'LM correction'])

    df = pd.DataFrame(index=index, columns=['Ø WER', 'Ø LER', 'Ø LER (raw)'])
    df.loc['greedy', 'lm_n']['Ø WER'] = df_metrics.loc['greedy', 'lm_n']['WER'].mean()
    df.loc['greedy', 'lm_y']['Ø WER'] = df_metrics.loc['greedy', 'lm_y']['WER'].mean()
    df.loc['beam', 'lm_n']['Ø WER'] = df_metrics.loc['beam', 'lm_n']['WER'].mean()
    df.loc['beam', 'lm_y']['Ø WER'] = df_metrics.loc['beam', 'lm_y']['WER'].mean()

    df.loc['greedy', 'lm_n']['Ø LER'] = df_metrics.loc['greedy', 'lm_n']['LER'].mean()
    df.loc['greedy', 'lm_y']['Ø LER'] = df_metrics.loc['greedy', 'lm_y']['LER'].mean()
    df.loc['beam', 'lm_n']['Ø LER'] = df_metrics.loc['beam', 'lm_n']['LER'].mean()
    df.loc['beam', 'lm_y']['Ø LER'] = df_metrics.loc['beam', 'lm_y']['LER'].mean()

    df.loc['greedy', 'lm_n']['Ø LER (raw)'] = df_metrics.loc['greedy', 'lm_n']['LER_raw'].mean()
    df.loc['greedy', 'lm_y']['Ø LER (raw)'] = df_metrics.loc['greedy', 'lm_y']['LER_raw'].mean()
    df.loc['beam', 'lm_n']['Ø LER (raw)'] = df_metrics.loc['beam', 'lm_n']['LER_raw'].mean()
    df.loc['beam', 'lm_y']['Ø LER (raw)'] = df_metrics.loc['beam', 'lm_y']['LER_raw'].mean()

    return df
