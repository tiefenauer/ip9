import argparse
import random
from datetime import timedelta
from operator import getitem
from os import listdir, makedirs, remove
from os.path import join, exists, getsize

import h5py
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from python_speech_features import mfcc
from scipy.io import wavfile
from tqdm import tqdm

from corpus.corpus import DeepSpeechCorpus
from util.audio_util import distort_audio
from util.corpus_util import get_corpus
from util.log_util import create_args_str

parser = argparse.ArgumentParser(description="""Export speech segments of corpus to CSV files and synthesize data""")
parser.add_argument('-id', type=str, required=True,
                    help='target-ID for processed files')
parser.add_argument('-s', '--source_dir', type=str, required=True,
                    help='id of corpus or path to corpus to export')
parser.add_argument('-t', '--target_dir', type=str, required=True,
                    help='target directory to save results')
parser.add_argument('-l', '--language', type=str, required=True,
                    help='language to use')
parser.add_argument('-f', '--force', action='store_true',
                    help='(optional) force override existing files. Default: False')
parser.add_argument('-x', '--synthesize', action='store_true',
                    help='whether to create synthesized data')
parser.add_argument('-num', '--include_numeric', action='store_true', default=False,
                    help='(optional) whether to include transcripts with numeric chars (default: False)')
parser.add_argument('-min', '--min_duration', nargs='?', type=int, default=0,
                    help='(optional) maximum number of speech segments minutes to process (default: all)')
parser.add_argument('-max', '--max_duration', nargs='?', type=int, default=0,
                    help='(optional) maximum number of speech segments minutes to process (default: all)')
parser.add_argument('-p', '--precompute_features', action='store_true',
                    help='(optional) precompute MFCC features in HDF5 format. Default: False')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))

    target_dir, corpus_id, force, synthesize, min_dur, max_dur, precompute_features = setup(args)

    corpus = get_corpus(args.source_dir, args.language)
    corpus.summary()

    print(f'processing {corpus.name} corpus and saving split segments in {target_dir}')
    csv_train, csv_dev, csv_test = extract_segments(target_dir, corpus_id, corpus, synthesize, min_dur, max_dur, force)
    print(f'done! All files are in {target_dir}')

    corpus = DeepSpeechCorpus(args.language, csv_train, csv_dev, csv_test)
    corpus.summary()

    if precompute_features:
        print(f'pre-computing features')
        compute_features(csv_train, csv_dev, csv_test, target_dir, force)


def setup(args):
    target_dir = join(args.target_dir, args.id)
    if not exists(target_dir):
        print(f'target directory {target_dir} does not exist. Creating...')
        makedirs(target_dir)

    force = args.force
    if not force and listdir(target_dir):
        inp = input(f"""
        WARNING: target directory {target_dir} already exists. Override?
        (this will overwrite all existing files in {target_dir} with the same names!!!) (Y/n)
        """)
        force = inp.lower() in ['', 'y']

    return target_dir, args.id, force, args.synthesize, args.min_duration, args.max_duration, args.precompute_features


def extract_segments(target_dir, corpus_id, corpus, synthesize=False, min_dur=0, max_dur=0, force=False):
    train_set = corpus.train_set(numeric=args.include_numeric)
    dev_set = corpus.dev_set(numeric=args.include_numeric)
    test_set = corpus.test_set(numeric=args.include_numeric)

    print(f'training length is: {timedelta(seconds=sum(seg.duration for seg in train_set))}')
    print(f'dev length is: {timedelta(seconds=sum(seg.duration for seg in dev_set))}')
    print(f'test length is: {timedelta(seconds=sum(seg.duration for seg in test_set))}')

    print(f'processing training segments')
    csv_train = process_subset('train', train_set, synthesize, corpus_id, target_dir, min_dur, max_dur, force)

    print(f'processing validation segments (data is only synthesized for training set)')
    csv_dev = process_subset('dev', dev_set, False, corpus_id, target_dir, min_dur, max_dur, force)

    print(f'processing validation segments (data is only synthesized for training set)')
    csv_test = process_subset('test', test_set, False, corpus_id, target_dir, min_dur, max_dur, force)

    return csv_train, csv_dev, csv_test


def process_subset(subset_id, subset, synthesize, corpus_id, target_dir, min_dur, max_dur, force):
    df = split_speech_segments(subset, corpus_id, subset_id, target_dir, synthesize, min_dur, max_dur, force)

    csv_path = join(target_dir, f'{corpus_id}-{subset_id}.csv')
    print(f'saving metadata in {csv_path}')
    df.to_csv(csv_path, index=False)
    return csv_path


def split_speech_segments(subset, corpus_id, subset_id, target_dir, synthesize, min_dur, max_dur, force):
    total = len(subset)
    if max_dur:
        print(f'trying to cap numer of speech segments to a total length of {max_dur} minutes. '
              f'Speech segements will be sorted by length before capping.')
        tot_duration = sum(s.duration for s in subset) / 60
        if tot_duration < max_dur:
            print(f'WARNING: maximum length of corpus was set to {max_dur} minutes, but total length of all '
                  f'speech segments is only {tot_duration} minutes! '
                  f'-> using all entries from corpus ({total} speech segments)')
        else:
            for i, s in enumerate(sorted(subset, key=lambda s: s.duration)):
                if sum(s.duration for s in subset[:i]) > max_dur * 60:
                    break
            print(f'total length of corpus will be capped at {max_dur} minutes ({i} speech segments)')
            total = i
            subset = subset[:i]

    segments = []
    files = []
    sum_duration = 0
    progress = tqdm(subset, total=total, unit=' speech segments')
    for i, segment in enumerate(progress):
        segment_id = f'{corpus_id}-{subset_id}-{i:0=4d}'
        wav_path = f'{segment_id}.wav'
        wav_path_absolute = join(target_dir, wav_path)

        if not exists(wav_path_absolute) or not getsize(wav_path_absolute) or force:
            sf.write(wav_path_absolute, segment.audio, segment.rate, subtype='PCM_16')

        segments.append((segment_id, segment.audio, segment.rate, segment.transcript))
        files.append((wav_path, getsize(wav_path_absolute), segment.duration, segment.transcript))
        sum_duration += segment.duration

        if synthesize:
            audio, rate = librosa.load(wav_path_absolute, sr=16000, mono=True)

            wav_shift = f'{segment_id}-shift.wav'
            wav_echo = f'{segment_id}-echo.wav'
            wav_high = f'{segment_id}-high.wav'
            wav_low = f'{segment_id}-low.wav'
            wav_fast = f'{segment_id}-fast.wav'
            wav_slow = f'{segment_id}-slow.wav'
            wav_loud = f'{segment_id}-loud.wav'
            wav_quiet = f'{segment_id}-quiet.wav'

            shift = random.uniform(0.5, 1.5)
            wav_shift_path = join(target_dir, wav_shift)
            wav_shift_len = synthesize_and_write(audio, rate, wav_shift_path, shift=shift, force=force)
            files.append((wav_shift, getsize(wav_shift_path), wav_shift_len, segment.transcript))

            echo = random.randint(30, 100)
            wav_echo_path = join(target_dir, wav_echo)
            wav_echo_len = synthesize_and_write(audio, rate, wav_echo_path, echo=echo, force=force)
            files.append((wav_echo, getsize(wav_echo_path), wav_echo_len, segment.transcript))

            higher = random.uniform(1.5, 5)
            wav_high_path = join(target_dir, wav_high)
            wav_high_len = synthesize_and_write(audio, rate, wav_high_path, pitch=higher, force=force)
            files.append((wav_high, getsize(wav_high_path), wav_high_len, segment.transcript))

            lower = random.uniform(-5, -1.5)
            wav_low_path = join(target_dir, wav_low)
            wav_low_len = synthesize_and_write(audio, rate, wav_low_path, pitch=lower, force=force)
            files.append((wav_low, getsize(wav_low_path), wav_low_len, segment.transcript))

            faster = random.uniform(1.2, 1.6)
            wav_fast_path = join(target_dir, wav_fast)
            wav_fast_len = synthesize_and_write(audio, rate, wav_fast_path, tempo=faster, force=force)
            files.append((wav_fast, getsize(wav_fast_path), wav_fast_len, segment.transcript))

            slower = random.uniform(0.6, 0.8)
            wav_slow_path = join(target_dir, wav_slow)
            wav_slow_len = synthesize_and_write(audio, rate, wav_slow_path, tempo=slower, force=force)
            files.append((wav_slow, getsize(wav_slow_path), wav_slow_len, segment.transcript))

            louder = random.randint(5, 15)
            wav_loud_path = join(target_dir, wav_loud)
            wav_loud_len = synthesize_and_write(audio, rate, wav_loud_path, volume=louder, force=force)
            files.append((wav_loud, getsize(wav_loud_path), wav_loud_len, segment.transcript))

            quieter = random.randint(-15, 5)
            wav_quiet_path = join(target_dir, wav_quiet)
            wav_quiet_len = synthesize_and_write(audio, rate, wav_quiet_path, volume=quieter, force=force)
            files.append((wav_quiet, getsize(wav_quiet_path), wav_quiet_len, segment.transcript))

        description = wav_path
        if max_dur:
            description += f' {timedelta(seconds=sum_duration)}'
        progress.set_description(description)

        if max_dur and sum_duration > max_dur * 60:
            break

    sum_duration = sum(getitem(t, 2) for t in files)

    if synthesize or min_dur and sum_duration < min_dur * 60 or max_dur and sum_duration < max_dur * 60:
        print(f'total length: {timedelta(seconds=sum_duration)}')
        print(f'filling up with distorted data until {timedelta(minutes=1000)} is reached')
        i = 0
        while sum_duration < 1000 * 60:
            i += 1
            for segment_id, audio, rate, transcript in tqdm(segments, unit=' segments'):
                shift = random.uniform(0.5, 1.5)
                pitch = random.uniform(-5, 5)
                tempo = random.uniform(0.6, 1.6)
                volume = random.randint(-15, 15)
                echo = random.randint(30, 100)

                wav_distort = f'{segment_id}-distorted-{i}.wav'
                wav_distort_path = join(target_dir, wav_distort)
                wav_distort_len = synthesize_and_write(audio, rate, wav_distort_path, shift=shift, pitch=pitch,
                                                       tempo=tempo, volume=volume, echo=echo, force=force)
                files.append((wav_distort, getsize(wav_distort_path), wav_distort_len, transcript))
                sum_duration += wav_distort_len

                if sum_duration > 1000 * 60:
                    break

            print(f'total length: {timedelta(seconds=sum_duration)}')

    return pd.DataFrame(data=files, columns=['wav_filename', 'wav_filesize', 'wav_length', 'transcript']).sort_values(
        'wav_length')


def synthesize_and_write(audio, rate, wav_path, shift=0, pitch=0, tempo=1, volume=0, echo=0, force=False):
    audio_synth = distort_audio(audio, rate,
                                shift_s=shift,
                                pitch_factor=pitch,
                                tempo_factor=tempo,
                                volume=volume,
                                echo=echo)

    if not exists(wav_path) or not getsize(wav_path) or force:
        sf.write(wav_path, audio_synth, rate, subtype='PCM_16')

    return len(audio_synth) / rate


def compute_features(csv_train, csv_valid, csv_test, target_dir, force):
    df_train = pd.read_csv(csv_train)
    df_dev = pd.read_csv(csv_valid)
    df_test = pd.read_csv(csv_test)

    h5_file_path = join(target_dir, 'features_mfcc.h5')
    if exists(h5_file_path) and force:
        remove(h5_file_path)
    if not exists(h5_file_path):
        with h5py.File(h5_file_path) as h5_file:
            create_subset(h5_file, 'train', df_train)
            create_subset(h5_file, 'test', df_dev)
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


if __name__ == '__main__':
    main(args)
