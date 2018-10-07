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
from pydub.utils import mediainfo
from python_speech_features import mfcc
from sklearn.utils import shuffle

from util.ctc_util import encode


class BatchGenerator(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
        self.cur_index = 0

    def __len__(self):
        return self.n // self.batch_size  # number of batches

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns a generator for the dataset. Because for training the dataset is iterated over once per epoch, this
        function is an endless loop over set.
        :return:
        """
        while True:
            if not self.has_next():
                self.cur_index = 0

            ret = self.get_batch(self.cur_index)
            self.cur_index += 1
            return ret

    def has_next(self):
        return (self.cur_index + 1) * self.batch_size < self.n

    def get_batch(self, idx):
        index_array = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        features = self.extract_features(index_array)
        X = pad_sequences(features, dtype='float32', padding='post')
        X_lengths = np.array([feature.shape[0] for feature in features])

        labels = self.extract_labels(index_array)
        Y = pad_sequences([encode(label) for label in labels], padding='post', value=28)
        Y_lengths = np.array([len(label) for label in labels])

        inputs = {
            'the_input': X,
            'the_labels': Y,
            'input_length': X_lengths,
            'label_length': Y_lengths,
            'source_str': labels
        }

        outputs = {'ctc': np.zeros([self.batch_size])}

        return inputs, outputs

    @abstractmethod
    def shuffle_entries(self):
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, index_array):
        """
        Extract unpadded features for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of unpadded features (batch_size x num_timesteps x num_features)
        """
        raise NotImplementedError

    @abstractmethod
    def extract_labels(self, index_array):
        """
        Extract unpadded, unencoded labels for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of textual labels
        """
        """"""
        raise NotImplementedError


class CSVBatchGenerator(BatchGenerator):

    def __init__(self, csv_path, sort=False, n_batches=None, batch_size=16, num_minutes=None):
        df, total_audio_length = read_data_from_csv(csv_path=csv_path, sort=sort)
        df['wav_filename'] = df['wav_filename'].map(lambda wav_file: join(dirname(abspath(csv_path)), wav_file))

        if n_batches:
            df = df.head(n_batches * batch_size)
        elif num_minutes:
            # truncate dataset to first {num_minutes} minutes of audio data
            if num_minutes * 60 > total_audio_length:
                print(f"""WARNING: {num_minutes} minutes is longer than total length of the dataset ({timedelta(seconds=total_audio_length)})!
                Training will be done on the whole dataset.""")
            else:
                df = truncate_dataset(df, batch_size, num_minutes)

        avg_audio_length = df['wav_length'].mean()
        print(f'average audio length: {timedelta(seconds=avg_audio_length)}')

        self.wav_files = df['wav_filename'].tolist()
        self.transcripts = df['transcript'].tolist()
        self.wav_sizes = df['wav_filesize'].tolist()
        self.wav_lengths = df['wav_length'].tolist()

        super().__init__(n=len(df.index), batch_size=batch_size)
        del df

    def shuffle_entries(self):
        self.wav_files, self.transcripts, self.wav_sizes = shuffle(self.wav_files, self.transcripts, self.wav_sizes)

    def extract_features(self, index_array):
        return [extract_mfcc(wav_file) for wav_file in (self.wav_files[i] for i in index_array)]

    def extract_labels(self, index_array):
        return [self.transcripts[i] for i in index_array]


def read_data_from_csv(csv_path, sort=True, create_word_list=False):
    """
    Read data from CSV into DataFrame.
    :param csv_path: absolute path to CSV file
    :param sort: whether to sort the files by file size
    :param create_word_list: whether to create a list of unique words
    :return:
        df: pd.DataFrame containing the CSV data
        total_audio_length: integer representing the total length of the audio by summing over the 'wav_length' column
                            from the CSV. If such a column is not present, the value returned is math.inf
    """
    if not isfile(csv_path):
        print(f'ERROR: CSV file {csv_path} does not exist!', file=sys.stderr)
        exit(0)

    print(f'Reading samples from {csv_path}...', end='')
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    print(f'done! ({len(df.index)} samples', end='')

    total_audio_length = math.inf

    if 'wav_length' in df:
        total_audio_length = df['wav_length'].sum()
        print(f', {timedelta(seconds=total_audio_length)})')
    else:
        print(')')

    if create_word_list:
        df['transcript'].to_csv(join('lm', 'df_all_word_list.csv'), header=False, index=False)

    if sort:
        df = df.sort_values(by='wav_filesize', ascending=True)

    avg_trans_length = np.mean([len(trans) for trans in df['transcript']])
    print(f'average transcript length: {avg_trans_length}')
    return df.reset_index(drop=True), total_audio_length


def extract_mfcc(wav_file_path):
    fs, audio = wav.read(wav_file_path)
    return mfcc(audio, samplerate=fs, numcep=26)  # (num_timesteps x num_features)


def truncate_dataset(df, batch_size, num_minutes):
    clip_ix = 0
    clipped_wav_length = 0.0
    if 'wav_length' in df:
        while (clipped_wav_length < num_minutes * 60  # total length must be required length
               and clip_ix < len(df.index)  # don't use more samples than available
               or clip_ix < batch_size):  # but use at least enough samples to fill a batch
            clip_ix += 1
            clipped_wav_length = sum(df['wav_length'][:clip_ix])
    else:
        print(f'length of WAV files not present in CSV file. Appending that information (this can take a while)...')
        df['wav_length'] = pd.Series(index=df.index)
        wav_lengths = []
        while clipped_wav_length < num_minutes * 60:
            wav_length = mediainfo(df.loc[clip_ix, 'wav_filename'])['duration']
            wav_lengths.append(float(wav_length))
            clipped_wav_length = sum(wav_lengths)
            clip_ix += 1
        df.loc[:clip_ix - 1, 'wav_length'] = wav_lengths

    print(f'clipping to first {clip_ix} samples ({timedelta(seconds=clipped_wav_length)}).')
    if clipped_wav_length > num_minutes * 60 and clip_ix == batch_size:
        print(f'length of clipped dataset is larger than {num_minutes} to fill at least 1 batch')
    return df.head(clip_ix)
