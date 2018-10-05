from os import makedirs
from os.path import exists, join, expanduser
from shutil import rmtree

import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras_preprocessing.image import Iterator
from keras_preprocessing.sequence import pad_sequences
from util.brnn_util import ctc_dummy_loss, decoder_dummy_loss, ler

from core.models import deep_speech_lstm
from core.report_callback import ReportCallback
from util.audio_util import shift
from util.corpus_util import get_corpus
from util.rnn_util import encode

target_dir = join(expanduser("~"), 'poc')


def main():
    print(f'All results and artifacts will be written to: {target_dir}')
    if exists(target_dir):
        print(f'removing target dir: {target_dir}')
        rmtree(target_dir)
    print(f'creating target dir: {target_dir}')
    makedirs(target_dir)

    corpus = get_corpus('ls')
    sample = corpus[0].speech_segments[0]
    print(f'training on a single sample with length {sample.audio_length}s')
    print(f'transcription: {sample.text}')

    train_gen = SampleGenerator(sample, lambda sample: sample.mfcc())

    audio = sample.audio

    def shift_audio_then_mfcc(sample):
        sample._audio = shift(audio)
        return sample.mfcc()

    val_gen = SampleGenerator(sample, shift_audio_then_mfcc)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "3"
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    model = deep_speech_lstm()
    model.summary()

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(loss={'ctc': ctc_dummy_loss, 'decoder': decoder_dummy_loss},
                  optimizer=opt,
                  metrics={'decoder': ler},  #
                  loss_weights=[1, 0]  # only optimize CTC cost
                  )

    report_cb = ReportCallback(model, val_gen, target_dir)
    model.fit_generator(train_gen, validation_data=val_gen, epochs=10000, callbacks=[report_cb])


class SampleGenerator(Iterator):

    def __init__(self, sample, features_fun):
        super().__init__(n=1, batch_size=1, shuffle=False, seed=0)
        self.sample = sample
        features = features_fun(sample)

        self.X = pad_sequences([features], dtype='float32', padding='post')
        self.X_lengths = np.array([features.shape[0]])

        self.Y_lengths = np.array([len(sample.text)])

        labels_encoded = [encode(sample.text)]
        rows, cols, data = [], [], []
        for row, label in enumerate(labels_encoded):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        self.Y = scipy.sparse.coo_matrix((data, (rows, cols)), dtype=np.int16)

    def _get_batches_of_transformed_samples(self, index_array):
        return [self.X, self.Y, self.X_lengths, self.Y_lengths], [np.zeros((self.X.shape[0],)), self.Y]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        index_array.sort()
        index_array_lst = index_array.tolist()
        return self._get_batches_of_transformed_samples(index_array_lst)


if __name__ == '__main__':
    main()
