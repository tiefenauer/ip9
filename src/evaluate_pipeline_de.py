import argparse
import os

from pattern3.metrics import levenshtein_similarity

from corpus.alignment import Voice
from util.string_util import normalize

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from os import makedirs
from os.path import abspath, exists, join, splitext, basename

import numpy as np
import pandas as pd
from keras import backend as K

from pipeline import pipeline
from util.corpus_util import get_corpus
from util.lm_util import load_lm, load_vocab
from util.log_util import create_args_str
from util.pipeline_util import query_lm_params, calculate_stats, create_demo_files, \
    update_index
from util.rnn_util import query_gpu
from util.visualization_util import visualize_pipeline_performance

parser = argparse.ArgumentParser(description="""
    Evaluate the pipeline using German samples from the ReadyLingua corpus. Because there is no reference model for
    German, the VAD stage is skipped and segmentation information is taken from corpus metadata instead.
    """)
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
parser.add_argument('--force_realignment', type=bool, action='store_true',
                    help='force realignment of partial transcript with original transcript, even if alignment'
                         'information is available from previous runs.')
parser.add_argument('--align_endings', type=bool, action='store_true',
                    help='align endings of partial transcripts, not just beginnings. If set to True, transcript may'
                         'contain unaligned parts between alignments. If set to False, each alignment ends where the'
                         'next one starts.')
parser.add_argument('--norm_transcript', type=bool, default=False,
                    help='Normalize transcript before alignment. If set to True, the alignments will be more accurate'
                         'because the transcript does not contain any punctuation, annotations and other clutter. '
                         'However, this might not reflect how the pipeline will be used. If set to False, the '
                         'partial transcripts will be aligned will be aligned with the original transcript as-is, '
                         'resulting in possibly less accurate alignments, but the original transcript will not be '
                         'changed')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    target_dir, keras_path, lm_path, vocab_path, gpu = setup(args)
    print(f'all results will be written to {target_dir}')

    lm = load_lm(lm_path) if lm_path else None
    vocab = load_vocab(vocab_path) if vocab_path else None

    corpus = get_corpus('rl', 'de')
    corpus.summary()
    test_entries = list(set((segment.entry for segment in corpus.test_set())))
    # add 6 entries from PodClub corpus
    corpus = get_corpus('pc', 'de')
    corpus.summary()
    test_entries += [corpus['record1058'], corpus['record1063'], corpus['record1076'], corpus['record1523'],
                     corpus['record1548'], corpus['record1556']]
    stats = []
    for i, entry in enumerate(test_entries):
        print(f'entry {i + 1}/{len(test_entries)}')
        audio_file = entry.audio_path
        sample_rate = entry.rate
        with open(entry.transcript_path, encoding='utf-8') as f:
            transcript = f.read()
            if args.norm_transcript:
                transcript = normalize(transcript)

        demo_id = splitext(basename(audio_file))[0]
        target_dir_entry = join(target_dir, demo_id)
        if not exists(target_dir_entry):
            makedirs(target_dir_entry)
        alignments_csv = join(target_dir_entry, 'alignments.csv')

        if target_dir and exists(alignments_csv):
            print(f'found inferences from previous run in {alignments_csv}')
            df_alignments = pd.read_csv(alignments_csv, header=0, index_col=0).replace(np.nan, '')
        else:
            print(f'Running pipeline using Keras model at {keras_path}, saving results in {target_dir_entry}')
            voiced_segments = [Voice(s.audio, s.rate, s.start_frame, s.end_frame) for s in entry]
            df_alignments = pipeline(voiced_segments=voiced_segments, sample_rate=sample_rate, transcript=transcript,
                                     language='de',
                                     keras_path=keras_path, lm=lm, vocab=vocab,
                                     force_realignment=args.force_realignment, align_endings=args.align_endings,
                                     target_dir=target_dir_entry)
            if target_dir:
                print(f'saving alignments to {alignments_csv}')
                df_alignments.to_csv(join(target_dir, alignments_csv))

        df_alignments.replace(np.nan, '', regex=True, inplace=True)

        df_stats = calculate_stats(df_alignments, keras_path, transcript)
        create_demo_files(target_dir_entry, audio_file, transcript, df_alignments, df_stats)

        # calculate average similarity between Keras-alignment and original aligments
        keras_alignments = df_alignments['alignment'].values
        original_alignments = [s.transcript for s in entry.segments]
        av_similarity = np.mean(
            [levenshtein_similarity(ka, oa) for (ka, oa) in zip(keras_alignments, original_alignments)])
        df_stats['similarity'] = av_similarity
        stats.append(df_stats)

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
