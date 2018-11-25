import argparse
from glob import glob
from itertools import chain
from os import makedirs
from os.path import abspath, splitext, exists, basename, join

import pandas as pd
from keras import backend as K
from pattern3.metrics import levenshtein_similarity

from pipeline import pipeline
from util.corpus_util import get_corpus
from util.lm_util import load_lm, load_vocab
from util.log_util import create_args_str
from util.pipeline_util import query_asr_params, calculate_stats, create_demo_files, query_lm_params, update_index
from util.rnn_util import query_gpu, load_keras_model, load_ds_model, create_tf_session
from util.visualization_util import visualize_pipeline_performance

parser = argparse.ArgumentParser(description="""
    Evaluate the performance of a pipeline by calculating the following values for each entry in a test set:
    - C: length of unaligned text (normalized by dividing by total length of ground truth)
    - O: length of overlapping alignments (normalized by dividing by total legnth of all alignments)
    - D: average Levenshtein distance over all alignments of the current entry

    The values are averaged through division by the number of entries in the test set.
    """)
parser.add_argument('--source_dir', type=str, required=False,
                    help=f'Path to directory containing pairs of audio files and transcripts to evaluate on.'
                    f'The audio format must be either WAV or MP3. The transcript must be a text file.'
                    f'Apart from the file extension, both audio and transcript file must have the same name to '
                    f'be identified as a pair.'
                    f'Either this or the --corpus argument must be set.')
parser.add_argument('--corpus', type=str, required=False,
                    help=f'Corpus path or ID to use for evaluation. If set, this will override the --source_dir '
                    f'argument. The elements from the training set of the respective corpus will be used for '
                    f'evaluation. Either this or the --source_dir argument must be set.')
parser.add_argument('--language', type=str, required=False,
                    help='(optional) language to use.  Only considered in conjunction with --corpus')
parser.add_argument('--target_dir', type=str, required=False,
                    help=f'Path to target directory where results will be written. '
                    f'If not set, the source directory will be used.')
parser.add_argument('--keras_path', type=str, required=True,
                    help=f'Path to root directory where Keras model is stored (*.h5 file).')
parser.add_argument('--ds_path', type=str, required=True,
                    help=f'(optional) Path to pre-trained DeepSpeech model (*.pbmm file).')
parser.add_argument('--ds_alpha_path', type=str, required=True,
                    help=f'(optional) Path to text file containing alphabet of DeepSpeech model.')
parser.add_argument('--ds_trie_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing trie for DeepSpeech model. '
                    f'Required if --ds_path is set')
parser.add_argument('--lm_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing KenLM n-gram Language Model')
parser.add_argument('--vocab_path', type=str, required=False,
                    help=f'(optional) Path to vocabulary for LM')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    demo_files, target_dir, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, vocab_path, gpu = setup(args)
    num_files = len(demo_files)
    print(f'Processing {num_files} audio/transcript samples. All results will be written to {target_dir}')

    lm = load_lm(lm_path) if lm_path else None
    vocab = load_vocab(vocab_path) if vocab_path else None

    stats_keras, stats_ds = [], []
    for i, (audio_file, transcript_file) in enumerate(demo_files):
        print(f'{i}/{num_files}: Evaluating pipeline on {audio_file}')
        run_id = splitext(basename(audio_file))[0]
        target_dir_ds = join(target_dir, run_id + '_ds')
        if not exists(target_dir_ds):
            makedirs(target_dir_ds)
        print(f'Using DS model at {ds_path}, saving results in {target_dir_ds}')
        print('-----------------------------------------------------------------')
        df_alignments_ds, transcript, language = pipeline(audio_file,
                                                          transcript_file=transcript_file,
                                                          ds_path=ds_path,
                                                          ds_alpha_path=ds_alpha_path,
                                                          ds_trie_path=ds_trie_path,
                                                          lm_path=lm_path,
                                                          target_dir=target_dir_ds)
        df_stats_ds = calculate_stats(df_alignments_ds, ds_path, transcript)
        create_demo_files(target_dir_ds, audio_file, transcript, df_alignments_ds, df_stats_ds)

        target_dir_keras = join(target_dir, run_id + '_keras')
        if not exists(target_dir_keras):
            makedirs(target_dir_keras)
        print(f'Using Keras model at {keras_path}, saving results in {target_dir_keras}')
        print('-----------------------------------------------------------------')
        df_alignments_keras, transcript, language = pipeline(audio_file,
                                                             transcript_file=transcript_file,
                                                             keras_path=keras_path,
                                                             lm=lm, vocab=vocab,
                                                             target_dir=target_dir_keras)
        df_stats_keras = calculate_stats(df_alignments_keras, keras_path, transcript)
        create_demo_files(target_dir_keras, audio_file, transcript, df_alignments_keras, df_stats_keras)
        print('-----------------------------------------------------------------')

        # average similarity between Keras and DeepSpeech alignments
        av_similarity = df_alignments_keras.join(df_alignments_ds, lsuffix='_keras', rsuffix='_ds')[
            ['alignment_keras', 'alignment_ds']] \
            .apply(lambda x: levenshtein_similarity(x[0], x[1]), axis=1) \
            .mean()

        for ix, row in df_stats_keras.iterrows():
            stats_keras.append(row.tolist() + [av_similarity])

        for ix, row in df_stats_ds.iterrows():
            stats_ds.append(row.tolist() + [av_similarity])

    columns = ['model path', 'transcript length', 'precision', 'recall', 'f-score', 'LER', 'similarity']
    df_keras = pd.DataFrame(stats_keras, columns=columns)
    csv_keras = join(target_dir, 'performance_keras.csv')
    df_keras.to_csv(csv_keras)

    df_ds = pd.DataFrame(stats_ds, columns=columns)
    csv_ds = join(target_dir, 'performance_ds.csv')
    df_ds.to_csv(csv_ds)
    print(f'summary saved to {csv_keras}')

    visualize_pipeline_performance(csv_keras, csv_ds, silent=True)
    update_index(target_dir, lang='en', num_aligned=len(demo_files),
                 df_keras=df_keras, keras_path=keras_path,
                 ds_path=ds_path, df_ds=df_ds,
                 lm_path=lm_path, vocab_path=vocab_path)
    K.clear_session()


def setup(args):
    if not args.source_dir and not args.corpus:
        raise ValueError('ERROR: Either --source_dir or --corpus must be set!')

    if args.corpus and not args.target_dir:
        raise ValueError('ERROR: If --corpus is set the --target_dir argument must be set!')

    if args.corpus:
        corpus = get_corpus(args.corpus, args.language)
        demo_files = [(entry.audio_path, entry.transcript_path) for entry in set(s.entry for s in corpus.test_set())]
        target_dir = abspath(args.target_dir)
    else:
        source_dir = abspath(args.source_dir)
        target_dir = abspath(args.target_dir) if args.target_dir else source_dir
        demo_files = []
        for audio_file in chain.from_iterable(glob(e) for e in (f'{source_dir}/*.{ext}' for ext in ('mp3', 'wav'))):
            transcript_file = splitext(audio_file)[0] + '.txt'
            if exists(transcript_file):
                print(f'adding: {basename(audio_file)} / {basename(transcript_file)}')
                demo_files.append((audio_file, transcript_file))

    keras_path, ds_path, ds_alpha_path, ds_trie_path = query_asr_params(args)
    lm_path, vocab_path = query_lm_params(args)
    gpu = args.gpu if args.gpu else query_gpu()

    return demo_files, target_dir, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, vocab_path, gpu


if __name__ == '__main__':
    main(args)
