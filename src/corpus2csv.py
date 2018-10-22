import argparse
from datetime import timedelta
from os import listdir, makedirs, remove
from os.path import join, exists, getsize

import h5py
import numpy as np
import pandas
import soundfile as sf
from python_speech_features import mfcc
from scipy.io import wavfile
from tqdm import tqdm

from util.corpus_util import get_corpus
from util.log_util import create_args_str

parser = argparse.ArgumentParser(description="""Export speech segments of corpus to CSV and split audio files""")
parser.add_argument('-id', type=str, help='target-ID for processed files')
parser.add_argument('-s', '--source_dir', type=str, help='id of corpus or path to corpus to export')
parser.add_argument('-t', '--target_dir', type=str, required=True, help='target directory to save results')
parser.add_argument('-l', '--language', type=str, help='language to use')
parser.add_argument('-num', '--include_numeric', action='store_true', default=False,
                    help='(optional) whether to include transcripts with numeric chars (default: False)')
parser.add_argument('-m', '--max', nargs='?', type=int, default=0,
                    help='(optional) maximum number of speech segments minutes to process (default: all)')
parser.add_argument('-f', '--force', action='store_true',
                    help='(optional) force override existing files. Default: False')
parser.add_argument('-p', '--precompute_features', action='store_true',
                    help='(optional) precompute MFCC features in HDF5 format. Default: False')
args = parser.parse_args()


def main():
    print(create_args_str(args))

    target_dir = join(args.target_dir, args.id)

    if not exists(target_dir):
        print(f'target directory {target_dir} does not exist. Creating...')
        makedirs(target_dir)

    override = False
    if not args.force and listdir(target_dir):
        inp = input(f"""
        WARNING: target directory {target_dir} already exists. Override?
        (this will overwrite all existing files in {target_dir} with the same names!!!) (Y/n)
        """)
        override = inp.lower() in ['', 'y']

    corpus = get_corpus(args.source_dir)(languages=args.language)
    corpus.summary()

    print(f'processing {corpus.name} corpus and saving split segments in {target_dir}')
    extract_speech_segments(args.id, corpus, target_dir, args.max, override, args.precompute_features)
    print(f'done! All files are in {target_dir}')


def extract_speech_segments(corpus_id, corpus, target_dir, max_audio_length, override=False, precompute_features=False):
    train_set, dev_set, test_set = corpus.train_dev_test_split(include_numeric=args.include_numeric)

    print(f'processing training segments')
    df_train = process_subset('train', train_set, corpus_id, target_dir, max_audio_length, override)

    print(f'processing validation segments')
    df_valid = process_subset('dev', dev_set, corpus_id, target_dir, max_audio_length * 0.2, override)

    print(f'processing validation segments')
    df_test = process_subset('test', test_set, corpus_id, target_dir, max_audio_length * 0.2, override)

    if precompute_features:
        print(f'pre-computing features')
        compute_features(df_train, df_valid, df_test, target_dir, override)


def process_subset(subset_id, subset, corpus_id, target_dir, max_audio_length, override):
    df = split_speech_segments(subset, corpus_id, subset_id, target_dir, max_audio_length, override)

    csv_path = join(target_dir, f'{corpus_id}-{subset_id}.csv')
    print(f'saving metadata in {csv_path}')
    df.to_csv(csv_path, index=False)
    return df


def split_speech_segments(subset, corpus_id, subset_id, target_dir, max_audio_length, override):
    files = []
    sum_audio_length = 0

    total = len(subset)

    if max_audio_length:
        print(f'trying to cap numer of corpus entry to a total length of {max_audio_length} minutes. '
              f'Speech segements will be sorted by length prior to capping.')
        tot_audio_length = sum([s.audio_length for s in subset]) / 60
        if tot_audio_length < max_audio_length:
            print(f'WARNING: maximum length of corpus was set to {max_audio_length} minutes, but total length of all '
                  f'speech segments is only {tot_audio_length} minutes! '
                  f'-> using all entries from corpus ({total} speech segments)')
        else:
            for i, s in enumerate(sorted(subset, key=lambda s: s.audio_length)):
                if sum([s.audio_length for s in subset[:i]]) > max_audio_length * 60:
                    break
            print(f'total length of corpus will be capped at {max_audio_length} minutes ({i} speech segments)')
            total = i

    progress = tqdm(subset, total=total, unit=' speech segments')
    for i, segment in enumerate(progress):
        segment_id = f'{corpus_id}-{subset_id}-{i:0=3d}'
        wav_path = f'{segment_id}.wav'
        txt_path = f'{segment_id}.txt'

        wav_path_absolute = join(target_dir, wav_path)
        txt_path_absolute = join(target_dir, txt_path)

        if not exists(wav_path_absolute) or not getsize(wav_path_absolute) or override:
            sf.write(wav_path_absolute, segment.audio, segment.rate, subtype='PCM_16')

        if not exists(txt_path_absolute) or not getsize(txt_path_absolute) or override:
            with open(txt_path_absolute, 'w') as f:
                transcript = f'{segment.start_frame} {segment.end_frame} {segment.text}'
                f.write(transcript)

        files.append((wav_path, getsize(wav_path_absolute), segment.audio_length, segment.text))
        sum_audio_length += segment.audio_length

        description = wav_path
        if max_audio_length:
            description += f' {timedelta(seconds=sum_audio_length)}'
        progress.set_description(description)

        if max_audio_length and sum_audio_length > max_audio_length * 60:
            break

    return pandas.DataFrame(data=files, columns=['wav_filename', 'wav_filesize', 'wav_length', 'transcript'])


def compute_features(df_train, df_valid, df_test, target_dir, override):
    h5_file_path = join(target_dir, 'features_mfcc.h5')
    if exists(h5_file_path) and override:
        remove(h5_file_path)
    if not exists(h5_file_path):
        with h5py.File(h5_file_path) as h5_file:
            create_subset(h5_file, 'train', df_train)
            create_subset(h5_file, 'test', df_valid)
            create_subset(h5_file, 'valid', df_test)


def create_subset(h5_file, name, df):
    h5_file.create_dataset(f'{name}/features', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.float32))
    h5_file.create_dataset(f'{name}/labels', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    h5_file.create_dataset(f'{name}/durations', shape=(0,), maxshape=(None,))
    progress = tqdm(zip(df['wav_filename'], df['wav_filesize'], df['transcript']), total=len(df.index))
    for wav_file_path, wav_file_size, transcript in progress:
        progress.set_description(f'{name}: {wav_file_path}')
        inputs = h5_file[name]['features']
        labels = h5_file[name]['labels']
        durations = h5_file[name]['durations']

        rate, audio = wavfile.read(wav_file_path)
        inp = mfcc(audio, samplerate=rate, numcep=26)  # (num_timesteps x num_features)
        inputs.resize(inputs.shape[0] + 1, axis=0)
        inputs[inputs.shape[0] - 1] = inp.flatten().astype(np.float32)

        labels.resize(labels.shape[0] + 1, axis=0)
        labels[labels.shape[0] - 1] = transcript

        durations.resize(durations.shape[0] + 1, axis=0)
        durations[durations.shape[0] - 1] = wav_file_size


def get_segment_id(segment):
    corpus_entry = segment.corpus_entry
    ix = corpus_entry.speech_segments.index(segment)
    return f'{corpus_entry.id}-{ix:0=3d}'


if __name__ == '__main__':
    main()
