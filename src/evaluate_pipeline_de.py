import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from os import makedirs
from os.path import abspath, exists, join, splitext, basename

import numpy as np
import pandas as pd
from keras import backend as K

from corpus.alignment import Voice
from pipeline import asr_keras, gsa
from util.corpus_util import get_corpus
from util.lm_util import load_lm, load_vocab
from util.log_util import create_args_str
from util.pipeline_util import create_alignments_dataframe, query_lm_params, calculate_stats, create_demo_files, \
    update_index
from util.rnn_util import query_gpu
from util.visualization_util import visualize_pipeline_performance

parser = argparse.ArgumentParser(description="""
    Evaluate the pipeline using German samples from the ReadyLingua corpus. Because there is no reference model for
    German, the VAD stage is skipped and the inference is only made 
    """)
parser.add_argument('--corpus', type=str, required=True,
                    help='corpus ID or path to corpus root directory')
parser.add_argument('--target_dir', type=str, required=False,
                    help=f'Path to target directory where results will be written. '
                    f'If not set, the source directory will be used.')
parser.add_argument('--keras_path', type=str, required=True,
                    help=f'(optional) Path to root directory where Keras model is stored (*.h5 file).'
                    f'If not set you will be asked at runtime.')
parser.add_argument('--lm_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing KenLM n-gram Language Model.'
                    f'If not set you will be asked at runtime.')
parser.add_argument('--vocab_path', type=str, required=False,
                    help=f'(optional) Path to vocabulary file to use for spell checker.'
                    f'If not set you will be asked at runtime.')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    target_dir, keras_path, lm_path, vocab_path, gpu = setup(args)
    print(f'all results will be written to {target_dir}')

    lm = load_lm(lm_path) if lm_path else None
    vocab = load_vocab(vocab_path) if vocab_path else None

    corpus = get_corpus(args.corpus, 'de')
    corpus.summary()
    test_entries = list(set((segment.entry for segment in corpus.test_set())))
    stats = []
    for i, entry in enumerate(test_entries):
        print(f'entry {i}/{len(test_entries)}')
        target_dir_entry = join(target_dir, splitext(basename(entry.audio_path))[0])
        if not exists(target_dir_entry):
            makedirs(target_dir_entry)
        alignments_csv = join(target_dir_entry, 'alignments.csv')

        if target_dir and exists(alignments_csv):
            print(f'found inferences from previous run in {alignments_csv}')
            df_alignments = pd.read_csv(alignments_csv, header=0, index_col=0).replace(np.nan, '')
        else:
            print(f'using simplified Keras model')
            voiced_segments = [Voice(s.audio, s.rate, s.start_frame, s.end_frame) for s in entry]
            transcripts = asr_keras(voiced_segments, 'de', entry.rate, keras_path, lm, vocab)
            df_alignments = create_alignments_dataframe(voiced_segments, transcripts, entry.rate)
            if target_dir:
                print(f'saving alignments to {alignments_csv}')
                df_alignments.to_csv(join(target_dir, alignments_csv))

        df_alignments.replace(np.nan, '', regex=True, inplace=True)

        if 'alignment' in df_alignments.keys():
            print(f'transcripts are already aligned')
        else:
            print(f'aligning transcript with {len(df_alignments)} transcribed voice segments')
            alignments = gsa(entry.transcript, df_alignments['transcript'].tolist())

            df_alignments['alignment'] = [a['text'] for a in alignments]
            df_alignments['text_start'] = [a['start'] for a in alignments]
            df_alignments['text_end'] = [a['end'] for a in alignments]

            print(f'saving alignments to {alignments_csv}')
            df_alignments.to_csv(alignments_csv)

        stats = calculate_stats(df_alignments, keras_path, entry.transcript)
        create_demo_files(target_dir_entry, entry.audio_path, entry.transcript, df_alignments, stats)
        stats.append(stats)

    df_keras = pd.concat(stats)
    csv_keras = join(target_dir, 'performance.csv')
    df_keras.to_csv(csv_keras)
    print(f'summary saved to {csv_keras}')

    visualize_pipeline_performance(csv_keras, csv_ds=None, silent=True)
    update_index(target_dir, lang='de', num_aligned=len(test_entries),
                 df_keras=df_keras, keras_path=keras_path,
                 lm_path=lm_path, vocab_path=vocab_path)
    K.clear_session()


def setup(args):
    target_dir = abspath(args.target_dir)
    if not exists(target_dir):
        makedirs(target_dir)

    while not args.keras_path:
        args.keras_path = input('Enter path to simplified Keras model to use for inference: ')
        if args.keras_path and not exists(abspath(args.keras_path)):
            raise ValueError(f'ERROR: Path {abspath(args.keras_path)} does not exist!')
    keras_path = abspath(args.keras_path)

    lm_path, vocab_path = query_lm_params(args)
    gpu = args.gpu if args.gpu else query_gpu()

    return target_dir, keras_path, lm_path, vocab_path, gpu


if __name__ == '__main__':
    main(args)
