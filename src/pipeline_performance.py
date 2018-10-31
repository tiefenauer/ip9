import argparse
from glob import glob
from itertools import chain
from os.path import abspath, splitext, exists, basename, join

import pandas as pd

from pipeline import pipeline
from util.corpus_util import get_corpus
from util.log_util import create_args_str
from util.pipeline_util import query_asr_params, calculate_stats, create_demo_files

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
                    help='(optional) language to use. Only consiedered in conjunction with --corpus')
parser.add_argument('--target_dir', type=str, required=False,
                    help=f'Path to target directory where results will be written. '
                         f'If not set, the source directory will be used.')
parser.add_argument('--keras_path', type=str, required=True,
                    help=f'(optional) Path to root directory where Keras model is stored (*.h5 file).')
parser.add_argument('--ds_path', type=str, required=True,
                    help=f'(optional) Path to pre-trained DeepSpeech model (*.pbmm file).')
parser.add_argument('--ds_alpha_path', type=str, required=True,
                    help=f'(optional) Path to text file containing alphabet of DeepSpeech model.')
parser.add_argument('--ds_trie_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing trie for DeepSpeech model. '
                         f'Required if --ds_path is set')
parser.add_argument('--lm_path', type=str, required=False,
                    help=f'(optional) Path to binary file containing KenLM n-gram Language Model')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    demo_files, target_dir, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, gpu = setup(args)
    print(f'processing {len(demo_files)}. All results will be written to {target_dir}')

    stats = []
    for i, (audio_file, transcript_file) in enumerate(demo_files):
        print(f'Evaluating pipeline on {audio_file}')
        run_id = splitext(basename(audio_file))[0]
        target_dir_ds = join(target_dir, run_id + '_ds')
        print(f'Using DS model at {ds_path}, saving results in {target_dir_ds}')
        print('-----------------------------------------------------------------')
        df_alignments_ds, transcript, language = pipeline(audio_file,
                                                          transcript_file=transcript_file,
                                                          ds_path=ds_path,
                                                          ds_alpha_path=ds_alpha_path,
                                                          ds_trie_path=ds_trie_path,
                                                          lm_path=lm_path,
                                                          target_dir=target_dir_ds)
        df_stats_ds = calculate_stats(df_alignments_ds)
        create_demo_files(target_dir_ds, audio_file, transcript, df_alignments_ds, df_stats_ds)

        target_dir_keras = join(target_dir, run_id + '_keras')
        print(f'Using Keras model at {keras_path}, saving results in {target_dir_keras}')
        print('-----------------------------------------------------------------')
        ds_alignments_keras, transcript, language = pipeline(audio_file,
                                                             transcript_file=transcript_file,
                                                             keras_path=keras_path,
                                                             lm_path=lm_path,
                                                             target_dir=target_dir_keras)
        df_stats_keras = calculate_stats(ds_alignments_keras)
        create_demo_files(target_dir_keras, audio_file, transcript, ds_alignments_keras, df_stats_keras)
        print('-----------------------------------------------------------------')

        for row in df_stats_ds.iterrows():
            stats.append([keras_path, len(transcript)] + row.tolist())

        for row in df_stats_keras.iterrows():
            stats.append([ds_path, len(transcript)] + row.tolist())

    df = pd.DataFrame(stats, columns=['engine', 'transcript_length', 'LER', 'similarity'])
    df.to_csv(join(target_dir, 'pipeline_performance.csv'))


def setup(args):
    if not args.source_dir and not args.corpus:
        raise ValueError('ERROR: Either --source_dir or --corpus must be set!')

    if args.corpus and not args.target_dir:
        raise ValueError('ERROR: If --corpus is set the --target_dir argument must be set!')

    if args.corpus:
        corpus = get_corpus(args.corpus, args.language)
        demo_files = [(entry.audio_path, entry.transcript_path) for entry in corpus.test_set()]
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

    keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, gpu = query_asr_params(args)
    return demo_files, target_dir, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, gpu


if __name__ == '__main__':
    main(args)
