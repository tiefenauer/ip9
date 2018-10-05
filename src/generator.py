import sys
from abc import abstractmethod
from datetime import timedelta
from genericpath import isfile
from os.path import join, dirname, abspath

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences
from python_speech_features import mfcc
from sklearn.utils import shuffle

from util.rnn_util import encode


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
        if n_batches:
            df = df.head(n_batches * batch_size)
        elif num_minutes:
            # truncate dataset to first {num_minutes} minutes of audio data
            if num_minutes * 60 > total_audio_length:
                print(f"""WARNING: {num_minutes} minutes is longer than total length of the dataset ({timedelta(seconds=total_audio_length)})!
                Training will be done on the whole dataset.""")
            else:
                clip_ix = 0
                batch_filled = False
                while (sum(df['wav_length'][:clip_ix]) < num_minutes * 60  # total length must be required length
                       and clip_ix < len(df.index)  # don't use more samples than available
                       or clip_ix < batch_size):  # but use at least enough samples to fill a batch
                    clip_ix += 1
                    if sum(df['wav_length'][:clip_ix]) > num_minutes * 60 and clip_ix == batch_size:
                        batch_filled = True
                df = df.head(clip_ix)
                audio_length = sum(df['wav_length'])
                print(f'clipped to first {clip_ix} samples ({timedelta(seconds=audio_length)} minutes).')
                if batch_filled:
                    print(f'total length is larger to fill at least 1 batch')

        csv_basedir = dirname(abspath(csv_path))
        self.wav_files = [join(csv_basedir, wav_file) for wav_file in df['wav_filename']]
        self.transcripts = [join(csv_basedir, trans_file) for trans_file in df['transcript']]
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
    if not isfile(csv_path):
        print(f'ERROR: CSV file {csv_path} does not exist!', file=sys.stderr)
        exit(0)

    print(f'Reading samples from {csv_path}...', end='')
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    total_audio_length = sum(df['wav_length'])
    print(f'done! ({len(df.index)} samples, {timedelta(seconds=total_audio_length)})')

    if create_word_list:
        df['transcript'].to_csv(join('lm', 'df_all_word_list.csv'), header=False, index=False)

    if sort:
        df = df.sort_values(by='wav_filesize', ascending=True)

    return df.reset_index(drop=True), total_audio_length


def extract_mfcc(wav_file_path):
    fs, audio = wav.read(wav_file_path)
    return mfcc(audio, samplerate=fs, numcep=26)  # (num_timesteps x num_features)
