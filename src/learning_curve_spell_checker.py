import argparse
from os.path import abspath, exists, join

import numpy as np
import pandas as pd
import seaborn as sns
from keras.optimizers import SGD
from matplotlib.cbook import mkdirs
from tqdm import tqdm

from corpus.alignment import Voice
from pipeline import asr_keras
from util.corpus_util import get_corpus
from util.lm_util import load_lm, load_vocab, correction, ler_norm, wer_norm
from util.log_util import create_args_str
from util.pipeline_util import query_lm_params
from util.rnn_util import query_gpu, load_keras_model

sns.set()
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True,
                    help=f'Corpus path or ID to use for evaluation. The elements from the validation will be used.')
parser.add_argument('--target_dir', type=str, required=False, help='root directory for all output.'
                                                                   'If not set you will be asked at runtime.')
parser.add_argument('--keras_dir', type=str, required=False, help='path to directory of Keras model to use.'
                                                                  'If not set you will be asked at runtime.')
parser.add_argument('--lm_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing KenLM n-gram Language Model.'
                    f'If not set you will be asked at runtime.')
parser.add_argument('--vocab_path', type=str, required=True, help='path to full vocabulary file')
parser.add_argument('--language', type=str, choices=['en', 'de', 'fr', 'it', 'es'], default='en',
                    help='language to train on. '
                         'English will use 26 characters from the alphabet, German 29 (umlauts)')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    corpus_id, language, target_dir, keras_dir, lm, vocab, gpu = setup(args)
    print(f'All results will be written to {target_dir}')

    predictions_csv = join(target_dir, 'predictions.csv')
    if exists(predictions_csv):
        print(f'loading cached transcriptions from {predictions_csv}')
        df = pd.read_csv(predictions_csv)
        ground_truths = df['ground_truth'].tolist()
        predictions = df['prediction'].tolist()
    else:
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        keras_model = load_keras_model(keras_dir, opt)
        corpus = get_corpus(corpus_id, language)
        voiced_segments = [Voice(s.audio, s.rate, s.start_frame, s.end_frame) for s in corpus.dev_set()]
        ground_truths = [s.transcript for s in corpus.dev_set()]
        predictions = asr_keras(voiced_segments, language, 16000, keras_model, lm, vocab)
        df = pd.DataFrame({'ground_truth': ground_truths, 'prediction': predictions})
        df.to_csv(predictions_csv)
        print(f'saved predictions to {predictions_csv}')

    tot = len(ground_truths)
    print(f'got {tot} ground truths and predictions')

    lers, wers = [], []
    index = list(range(1000, 200000, 1000))
    for i in index:
        print(f'Spell-checking with top {i} words...')
        vocab_reduced = vocab[:i]
        predictions_corr = [correction(t, language, lm, vocab_reduced) for t in tqdm(predictions)]
        print('calculating average LER')
        lers += np.mean(ler_norm(gt, p) for (gt, p) in tqdm(zip(ground_truths, predictions_corr), total=tot))
        print('calculating average WER')
        wers += np.mean(wer_norm(gt, p) for (gt, p) in tqdm(zip(ground_truths, predictions_corr), total=tot))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('# words in vocabulary')
    # ax.legend(['average LER', 'average_LER'])
    ax.set_title('Learning curve for spell checker')

    data = {'ler': lers, 'wer': wers}
    df = pd.DataFrame(data, index)
    df.plot(ax=ax)

    plot_csv = join(target_dir, 'plot.csv')
    fig.savefig(plot_csv)
    print(f'saved plot to {plot_csv}')


def setup(args):
    while not args.target_dir:
        args.target_dir = input('Enter target directory to write results to: ')
    target_dir = abspath(args.target_dir)
    if not exists(target_dir):
        mkdirs(target_dir)

    keras_dir = abspath(args.keras_dir)
    if not exists(keras_dir):
        raise ValueError(f'ERROR: model directory {keras_dir} does not exist!')

    gpu = query_gpu(args.gpu)
    lm_path, vocab_path = query_lm_params(args)
    lm = load_lm(lm_path)
    vocab = load_vocab(vocab_path)

    return args.corpus, args.language, target_dir, keras_dir, lm, vocab, gpu


if __name__ == '__main__':
    main(args)
