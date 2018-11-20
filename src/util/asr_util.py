import itertools
import math

import pandas as pd
from tqdm import tqdm

from util.lm_util import ler_norm, wer_norm, ler, correction, wer

decoding_strategies = ['greedy', 'beam']
lm_uses = ['lm_n', 'lm_40k', 'lm_80k', 'lm_160k']
metrics = ['WER', 'LER', 'WER_raw', 'LER_raw']


def infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, language, lm, vocab):
    batch_generator.cur_index = 0  # reset index
    batch_size = batch_generator.batch_size

    inferences = []
    for ix in tqdm(range(len(batch_generator)), desc='transcribing/decoding batch', unit=' batches', position=1):
        batch_inputs, _ = next(batch_generator)
        batch_inferences = infer_batch(ix, batch_size, batch_inputs, decoder_greedy, decoder_beam, language, lm, vocab)
        inferences.append(batch_inferences)

    df_inferences = pd.concat(inferences, sort=False)
    df_inferences.index.name = 'ground truth'
    return df_inferences


def infer_batch(batch_ix, batch_size, batch_inputs, decoder_greedy, decoder_beam, language, lm=None, lm_vocab=None):
    batch_input = batch_inputs['the_input']
    batch_input_lengths = batch_inputs['input_length']

    if 'source_str' in batch_inputs:
        ground_truths = batch_inputs['source_str']
    else:
        indexes = range(batch_ix * batch_size, batch_ix * batch_size + len(batch_input))
        ground_truths = [str(i) for i in indexes]

    preds_greedy = decoder_greedy.decode(batch_input, batch_input_lengths)
    preds_beam = decoder_beam.decode(batch_input, batch_input_lengths)

    preds_greedy_40k, preds_greedy_80k, preds_greedy_160k = [], [], []
    for pred_greedy in tqdm(preds_greedy, unit=' voice segments', desc='making corrections (greedy)', position=0):
        preds_greedy_40k.append(correction(pred_greedy, language, lm, lm_vocab.words[:40000]))
        preds_greedy_80k.append(correction(pred_greedy, language, lm, lm_vocab.words[:80000]))
        preds_greedy_160k.append(correction(pred_greedy, language, lm, lm_vocab.words[:160000]))

    preds_beam_40k, preds_beam_80k, preds_beam_160k = [], [], []
    for pred_beam in tqdm(preds_beam, unit=' voice segments', desc='making corrections (beam)', position=0):
        preds_beam_40k.append(correction(pred_beam, language, lm, lm_vocab.words[:40000]))
        preds_beam_80k.append(correction(pred_beam, language, lm, lm_vocab.words[:80000]))
        preds_beam_160k.append(correction(pred_beam, language, lm, lm_vocab.words[:160000]))

    columns = pd.MultiIndex.from_product([decoding_strategies, lm_uses, ['prediction'] + metrics],
                                         names=['decoding strategy', 'LM correction', 'predictions'])

    df = pd.DataFrame(index=ground_truths, columns=columns)
    for ground_truth, pred_greedy, pred_greedy_40k, pred_greedy_80k, pred_greedy_160k, pred_beam, pred_beam_40k, pred_beam_80k, pred_beam_160k in zip(
            ground_truths,
            preds_greedy, preds_greedy_40k, preds_greedy_80k, preds_greedy_160k,
            preds_beam, preds_beam_40k, preds_beam_80k, preds_beam_160k):
        df.loc[ground_truth, ('greedy', 'lm_n', 'prediction')] = pred_greedy
        df.loc[ground_truth, ('greedy', 'lm_n', 'WER')] = wer_norm(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'LER')] = ler_norm(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'WER_raw')] = wer(ground_truth, pred_greedy)
        df.loc[ground_truth, ('greedy', 'lm_n', 'LER_raw')] = ler(ground_truth, pred_greedy)

        df.loc[ground_truth, ('greedy', 'lm_40k', 'prediction')] = pred_greedy_40k
        df.loc[ground_truth, ('greedy', 'lm_40k', 'WER')] = wer_norm(ground_truth, pred_greedy_40k)
        df.loc[ground_truth, ('greedy', 'lm_40k', 'LER')] = ler_norm(ground_truth, pred_greedy_40k)
        df.loc[ground_truth, ('greedy', 'lm_40k', 'WER_raw')] = wer(ground_truth, pred_greedy_40k)
        df.loc[ground_truth, ('greedy', 'lm_40k', 'LER_raw')] = ler(ground_truth, pred_greedy_40k)

        df.loc[ground_truth, ('greedy', 'lm_80k', 'prediction')] = pred_greedy_80k
        df.loc[ground_truth, ('greedy', 'lm_80k', 'WER')] = wer_norm(ground_truth, pred_greedy_80k)
        df.loc[ground_truth, ('greedy', 'lm_80k', 'LER')] = ler_norm(ground_truth, pred_greedy_80k)
        df.loc[ground_truth, ('greedy', 'lm_80k', 'WER_raw')] = wer(ground_truth, pred_greedy_80k)
        df.loc[ground_truth, ('greedy', 'lm_80k', 'LER_raw')] = ler(ground_truth, pred_greedy_80k)

        df.loc[ground_truth, ('greedy', 'lm_160k', 'prediction')] = pred_greedy_160k
        df.loc[ground_truth, ('greedy', 'lm_160k', 'WER')] = wer_norm(ground_truth, pred_greedy_160k)
        df.loc[ground_truth, ('greedy', 'lm_160k', 'LER')] = ler_norm(ground_truth, pred_greedy_160k)
        df.loc[ground_truth, ('greedy', 'lm_160k', 'WER_raw')] = wer(ground_truth, pred_greedy_160k)
        df.loc[ground_truth, ('greedy', 'lm_160k', 'LER_raw')] = ler(ground_truth, pred_greedy_160k)

        df.loc[ground_truth, ('beam', 'lm_n', 'prediction')] = pred_beam
        df.loc[ground_truth, ('beam', 'lm_n', 'WER')] = wer_norm(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'LER')] = ler_norm(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'WER_raw')] = wer(ground_truth, pred_beam)
        df.loc[ground_truth, ('beam', 'lm_n', 'LER_raw')] = ler(ground_truth, pred_beam)

        df.loc[ground_truth, ('beam', 'lm_40k', 'prediction')] = pred_beam_40k
        df.loc[ground_truth, ('beam', 'lm_40k', 'WER')] = wer_norm(ground_truth, pred_beam_40k)
        df.loc[ground_truth, ('beam', 'lm_40k', 'LER')] = ler_norm(ground_truth, pred_beam_40k)
        df.loc[ground_truth, ('beam', 'lm_40k', 'WER_raw')] = wer(ground_truth, pred_beam_40k)
        df.loc[ground_truth, ('beam', 'lm_40k', 'LER_raw')] = ler(ground_truth, pred_beam_40k)

        df.loc[ground_truth, ('beam', 'lm_80k', 'prediction')] = pred_beam_80k
        df.loc[ground_truth, ('beam', 'lm_80k', 'WER')] = wer_norm(ground_truth, pred_beam_80k)
        df.loc[ground_truth, ('beam', 'lm_80k', 'LER')] = ler_norm(ground_truth, pred_beam_80k)
        df.loc[ground_truth, ('beam', 'lm_80k', 'WER_raw')] = wer(ground_truth, pred_beam_80k)
        df.loc[ground_truth, ('beam', 'lm_80k', 'LER_raw')] = ler(ground_truth, pred_beam_80k)

        df.loc[ground_truth, ('beam', 'lm_160k', 'prediction')] = pred_beam_160k
        df.loc[ground_truth, ('beam', 'lm_160k', 'WER')] = wer_norm(ground_truth, pred_beam_160k)
        df.loc[ground_truth, ('beam', 'lm_160k', 'LER')] = ler_norm(ground_truth, pred_beam_160k)
        df.loc[ground_truth, ('beam', 'lm_160k', 'WER_raw')] = wer(ground_truth, pred_beam_160k)
        df.loc[ground_truth, ('beam', 'lm_160k', 'LER_raw')] = ler(ground_truth, pred_beam_160k)

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
