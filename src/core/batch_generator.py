import math
import sys
from abc import abstractmethod
from datetime import timedelta
from genericpath import isfile
from os.path import join, dirname, abspath

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from pydub.utils import mediainfo
from python_speech_features import mfcc
from sklearn.utils import shuffle

from util.ctc_util import encode, get_tokens


class BatchGenerator(Sequence):
    """
    Generates batches of (input, output) tuples that can be fed to the simplified DeepSpeech model.
    A contains the following items:
        - input: MFCC features of an audio signal
        - input_length: the items in each batch must have equal length. The feature sequences are
          therefore zero-padded and the original length is stored.
        - the_labels: encoded label sequences for the audio signal, zero-padded (like the feature sequences)
        - label_length: length of each label sequence
        - source_str: the original, unencoded transcript (ground truth)
    """

    def __init__(self, batch_items, batch_size, lang):
        """
        Initialize BatchGenerator
        :param batch_items: sequence of anything. This can be a list, a DataFrame, ...
        :param batch_size: number of elements in each batch
        :param lang: language to use. This will affect the tokens used for encoding the labels
        """
        self.batch_items = batch_items
        self.batch_size = batch_size
        self.cur_index = 0
        self.tokens = get_tokens(lang)

    def __getitem__(self, idx):
        first = idx * self.batch_size
        last = (idx + 1) * self.batch_size
        features = self.extract_features(first, last)

        X = pad_sequences(features, dtype='float32', padding='post')
        X_lengths = np.array([feature.shape[0] for feature in features])

        inputs = {
            'the_input': X,
            'input_length': X_lengths
        }

        # labels are only known when inferring on labelled data (train-/dev-/test-set), not on completely new data
        labels = self.extract_labels(first, last)
        if labels:
            Y = pad_sequences([encode(label, self.tokens) for label in labels], padding='post', value=28)
            Y_lengths = np.array([len(label) for label in labels])
            inputs['the_labels'] = Y
            inputs['label_length'] = Y_lengths
            inputs['source_str'] = labels

        outputs = {'ctc': np.zeros([len(features)])}

        return inputs, outputs

    def __len__(self):
        """
        The length of a batch generator is equal to the number of batches it produces
        """
        return int(np.ceil(len(self.batch_items) / float(self.batch_size)))

    @abstractmethod
    def shuffle_entries(self):
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, first, last):
        """
        Extract unpadded features for a batch of elements with matching indices
        :param first: first index to process
        :param last: last index to process
        :return: list of unpadded features (batch_size x num_timesteps x num_features)
        """
        raise NotImplementedError

    @abstractmethod
    def extract_labels(self, first, last):
        """
        Extract unpadded, unencoded labels for a batch of elements with matching indices
        :param first: first index to process
        :param last: last index to process
        :return: list of textual labels
        """
        """"""
        raise NotImplementedError


class VoiceSegmentsBatchGenerator(BatchGenerator):
    """
    Generate batches from voiced segments
    """

    def __init__(self, voiced_segments, sample_rate, batch_size, lang):
        """
        Initialize the BatchGenerator
        :param voiced_segments: list of corpus.alignment.Voice objects
        :param sample_rate: sampling rate of the audio signal
        """
        super().__init__(voiced_segments, batch_size, lang)
        self.sample_rate = sample_rate

    def extract_features(self, first, last):
        return [mfcc(segment.audio, samplerate=self.sample_rate, numcep=26) for segment in self.batch_items[first:last]]

    def extract_labels(self, first, last):
        # no labels for VAD-recognized voiced segments
        return None

    def shuffle_entries(self):
        self.batch_items = shuffle(self.batch_items)


class CSVBatchGenerator(BatchGenerator):
    """
    Generate batches from CSV-based corpus
    """

    def __init__(self, csv_path, lang, sort=False, n_batches=None, batch_size=16, n_minutes=None, use_synth=False):
        """
        Initialize BatchGenerator
        :param csv_path: absolute path to CSV index file containing the segmentation information and transcripts
        :param sort: whether to sort the samples by audio length
        :param n_batches: maximum number of batches to generate. When set, only the first n batches are generated.
        :param n_minutes: maximum number of minutes to process. When set, only batches up to n minutes will be generated
        :param use_synth: whether to include synthesized data in the generated batches or not
        """
        df, duration = read_data_from_csv(csv_path=csv_path, sort=sort)
        if not use_synth:
            synth_suffixes = ['-high', '-low', '-fast', '-slow', '-loud', '-quiet', '-shift', '-echo', '-distorted']
            print(f'keeping only non-synthesized data (samples ending in {synth_suffixes}))')

            def is_original(row):
                return not any(suffix in row['wav_filename'] for suffix in synth_suffixes)

            df = df[df.apply(is_original, axis=1)]
            print(f'kept {len(df)} samples')

        df['wav_filename'] = df['wav_filename'].map(lambda wav_file: join(dirname(abspath(csv_path)), wav_file))

        if n_batches:
            df = df.head(n_batches * batch_size)
        elif n_minutes:
            # truncate dataset to first {num_minutes} minutes of audio data
            if n_minutes * 60 > duration:
                print(f"""
                WARNING: {n_minutes} minutes ({timedelta(seconds=n_minutes)}) is longer than 
                total length of the dataset ({timedelta(seconds=duration)})!
                Training will be done on the whole dataset.""")
            else:
                df = truncate_df(df, batch_size, n_minutes)

        if 'wav_length' in df:
            avg_duration = df['wav_length'].mean()
            print(f'average audio length: {timedelta(seconds=avg_duration)}')

        self.wav_files = df['wav_filename'].values
        self.transcripts = df['transcript'].values
        self.wav_sizes = df['wav_filesize'].values
        self.wav_lengths = df['wav_length'].values if 'wav_length' in df else np.empty()

        super().__init__(batch_items=df, batch_size=batch_size, lang=lang)
        del df

    def shuffle_entries(self):
        self.wav_files, self.transcripts, self.wav_sizes = shuffle(self.wav_files, self.transcripts, self.wav_sizes)

    def extract_features(self, first, last):
        return [extract_mfcc(wav_file) for wav_file in self.wav_files[first:last]]

    def extract_labels(self, first, last):
        return self.transcripts[first:last].tolist()


def read_data_from_csv(csv_path, sort=True):
    """
    Read data from CSV into DataFrame.
    :param csv_path: absolute path to CSV file
    :param sort: whether to sort the files by file size
    :return:
        df: pd.DataFrame containing the CSV data
        duration: integer representing the total length of the audio by summing over the 'wav_length' column
                  from the CSV. If such a column is not present, the value returned is math.inf
    """
    if not isfile(csv_path):
        print(f'ERROR: CSV file {csv_path} does not exist!', file=sys.stderr)
        exit(0)

    print(f'Reading samples from {csv_path}...', end='')
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    print(f'done! ({len(df.index)} samples', end='')

    duration = math.inf

    if 'wav_length' in df:
        duration = df['wav_length'].sum()
        avg_duration = df['wav_length'].mean()
        print(f', {timedelta(seconds=duration)}, Ø audio length: {avg_duration:.4f} seconds)')
    else:
        print(')')

    if sort:
        df = df.sort_values(by='wav_filesize', ascending=True)

    avg_trans_length = np.mean([len(trans) for trans in df['transcript']])
    print(f'average transcript length: {avg_trans_length}')
    return df.reset_index(drop=True), duration


def extract_mfcc(wav_file_path):
    fs, audio = wav.read(wav_file_path)
    return mfcc(audio, samplerate=fs, numcep=26)  # (num_timesteps x num_features)


def truncate_df(df, batch_size, n_minutes):
    """
    Truncate a dataset to contain only the first n minutes of audio. Samples from the DataFrame will be considered
    until their total length is higher than n minutes. The other samples are discarded.

    :param df: DataFrame containing the segmentation information (samples)
    :param batch_size: batch size to use. At least one batch will be filled, even if its total length is longer than n.
    :param n_minutes: number of minutes to include in the dataset
    :return: the first x rows (samples) of the DataFrame whose cumulative length is about n_minutes
    """
    clip_ix = 0
    clipped_wav_length = 0.0
    if 'wav_length' in df:
        while (clipped_wav_length < n_minutes * 60  # total length must be required length
               and clip_ix < len(df.index)  # don't use more samples than available
               or clip_ix < batch_size):  # but use at least enough samples to fill a batch
            clip_ix += 1
            clipped_wav_length = sum(df['wav_length'][:clip_ix])
    else:
        print(f'length of WAV files not present in CSV file. Appending that information (this can take a while)...')
        df['wav_length'] = pd.Series(index=df.index)
        wav_lengths = []
        while clipped_wav_length < n_minutes * 60:
            wav_length = mediainfo(df.loc[clip_ix, 'wav_filename'])['duration']
            wav_lengths.append(float(wav_length))
            clipped_wav_length = sum(wav_lengths)
            clip_ix += 1
        df.loc[:clip_ix - 1, 'wav_length'] = wav_lengths

    print(f'clipping to first {clip_ix} samples ({timedelta(seconds=clipped_wav_length)}).')
    if clipped_wav_length > n_minutes * 60 and clip_ix == batch_size:
        print(f'length of clipped dataset is larger than {n_minutes} to fill at least 1 batch')
    return df.head(clip_ix)
