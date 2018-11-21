import itertools
import math

import pandas as pd
from tqdm import tqdm

from util.lm_util import ler_norm, wer_norm, ler, correction, wer

decoding_strategies = ['greedy', 'beam']
lm_uses = ['lm_n', 'lm_y']
metrics = ['WER', 'LER', 'WER_raw', 'LER_raw']


def infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, language, lm, vocab):
    inferences = []
    for ix in tqdm(range(len(batch_generator)), desc='transcribing batch', unit=' batches', position=1):
        batch_inputs, _ = batch_generator[ix]
        x = batch_inputs['the_input']
        x_len = batch_inputs['input_length']
        if 'source_str' in batch_inputs:
            y = batch_inputs['source_str']
        else:
            y = [str(i) for i in range(ix * batch_generator.batch_size, ix * batch_generator.batch_size + len(x))]

        batch_inferences = infer_batch(x, x_len, y, decoder_greedy, decoder_beam,language, lm, vocab)
        inferences.append(batch_inferences)
        ix += 1

    df_inferences = pd.concat(inferences, sort=False)
    df_inferences.index.name = 'ground truth'
    return df_inferences


def infer_batch(x, x_len, y, decoder_greedy, decoder_beam, language, lm=None, lm_vocab=None):
    preds_greedy = decoder_greedy.decode(x, x_len)
    preds_beam = decoder_beam.decode(x, x_len)

    preds_greedy_lm = [correction(pred_greedy, language, lm, lm_vocab) for pred_greedy in
                       tqdm(preds_greedy, unit=' voice segments', desc='making corrections (greedy)', position=0)]
    preds_beam_lm = [correction(pred_beam, language, lm, lm_vocab) for pred_beam in
                     tqdm(preds_beam, unit=' voice segments', desc='making corrections (beam)', position=0)]

    columns = pd.MultiIndex.from_product([decoding_strategies, lm_uses, ['prediction'] + metrics],
                                         names=['decoding strategy', 'LM correction', 'predictions'])

    df = pd.DataFrame(index=y, columns=columns)
    for ground_truth, pred_greedy, pred_greedy_lm, pred_beam, pred_beam_lm in zip(y,
                                                                                  preds_greedy, preds_greedy_lm,
                                                                                  preds_beam, preds_beam_lm):
        df.loc[ground_truth, ('greedy', 'lm_n', 'prediction')] = pred_greedy
        df.loc[ground_truth, ('greedy', 'lm_n', 'WER')] = wer_norm(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'LER')] = ler_norm(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'WER_raw')] = wer(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'LER_raw')] = ler(ground_truth, pred_greedy)

        df.loc[ground_truth, ('greedy', 'lm_y', 'prediction')] = pred_greedy_lm
        df.loc[ground_truth, ('greedy', 'lm_y', 'WER')] = wer_norm(ground_truth, pred_greedy_lm)
        df.loc[ground_truth, ('greedy', 'lm_y', 'LER')] = ler_norm(ground_truth, pred_greedy_lm)
        df.loc[ground_truth, ('greedy', 'lm_y', 'WER_raw')] = wer(ground_truth, pred_greedy_lm)
        df.loc[ground_truth, ('greedy', 'lm_y', 'LER_raw')] = ler(ground_truth, pred_greedy_lm)

        df.loc[ground_truth, ('beam', 'lm_n', 'prediction')] = pred_beam
        df.loc[ground_truth, ('beam', 'lm_n', 'WER')] = wer_norm(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'LER')] = ler_norm(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'WER_raw')] = wer(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'LER_raw')] = ler(ground_truth, pred_beam)

        df.loc[ground_truth, ('beam', 'lm_y', 'prediction')] = pred_beam_lm
        df.loc[ground_truth, ('beam', 'lm_y', 'WER')] = wer_norm(ground_truth, pred_beam_lm)
        df.loc[ground_truth, ('beam', 'lm_y', 'LER')] = ler_norm(ground_truth, pred_beam_lm)
        df.loc[ground_truth, ('beam', 'lm_y', 'WER_raw')] = wer(ground_truth, pred_beam_lm)
        df.loc[ground_truth, ('beam', 'lm_y', 'LER_raw')] = ler(ground_truth, pred_beam_lm)

    return df


def calculate_metrics_mean(df_inferences):
    index = pd.MultiIndex.from_product([decoding_strategies, lm_uses], names=['decoding strategy', 'LM correction'])
    df = pd.DataFrame(index=index, columns=metrics)

    for decoding_strategy, lm_used, metric in itertools.product(decoding_strategies, lm_uses, metrics):
        df.loc[decoding_strategy, lm_used][metric] = df_inferences[decoding_strategy, lm_used, metric].mean()

    return df


def extract_best_transcript(df_inferences):
    transcripts = [''] * len(df_inferences)

    for ix, row in df_inferences.iterrows():
        ler_min = math.inf
        transcript = ''
        for decoding_strategy, lm_use in itertools.product(decoding_strategies, lm_uses):
            ler_value = row[(decoding_strategy, lm_use)]['LER_raw']
            if ler_value < ler_min:
                ler_min = ler_value
                transcript = row[(decoding_strategy, lm_use)]['prediction']
        transcripts[int(ix)] = transcript
    return transcripts
