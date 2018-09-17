from os import makedirs
from os.path import exists, join, expanduser
from shutil import rmtree

import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras_preprocessing.image import Iterator
from keras_preprocessing.sequence import pad_sequences
from python_speech_features import mfcc

from core.callbacks import ReportCallback
from util.audio_util import shift
from util.brnn_util import deep_speech_model, ctc_dummy_loss, decoder_dummy_loss, ler
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

    train_gen = SampleGenerator(sample, shift_audio=False)
    val_gen = SampleGenerator(sample, shift_audio=True)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "3"
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    model = deep_speech_model(num_features=26)
    model.summary()

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(optimizer=opt, loss=ctc_dummy_loss)

    report_cb = ReportCallback(model, val_gen, target_dir)
    tb_cb = TensorBoard(log_dir=target_dir, write_graph=False, write_images=True)
    callbacks = [report_cb, tb_cb]
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=10000,
                        callbacks=callbacks,
                        initial_epoch=0,
                        verbose=1,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1)


class SampleGenerator(Iterator):

    def __init__(self, sample, shift_audio=False):
        super().__init__(n=1, batch_size=1, shuffle=False, seed=0)
        self.sample = sample
        self.shift_audio = shift_audio

    def _get_batches_of_transformed_samples(self, index_array):
        if self.shift_audio:
            audio = shift(self.sample.audio)
        else:
            audio = self.sample.audio

        features = mfcc(audio, self.sample.rate, numcep=26)

        self.X = pad_sequences([features], dtype='float32', padding='post')
        self.X_lengths = np.array([features.shape[0]])

        self.Y_lengths = np.array([len(self.sample.text)])

        self.source_str = np.array([self.sample.text])

        labels_encoded = [encode(self.sample.text)]
        rows, cols, data = [], [], []
        for row, label in enumerate(labels_encoded):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        self.Y = scipy.sparse.coo_matrix((data, (rows, cols)), dtype=np.int16)
        inputs = {
            'the_input': self.X,
            'the_labels': self.Y,
            'input_length': self.X_lengths,
            'label_length': self.Y_lengths,
            'source_str': self.source_str
        }
        outputs = {
            'ctc': np.zeros([self.X.shape[0]])
        }
        return inputs, outputs
        # return [self.X, self.Y, self.X_lengths, self.Y_lengths], [np.zeros((self.X.shape[0],)), self.Y]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        index_array.sort()
        index_array_lst = index_array.tolist()
        return self._get_batches_of_transformed_samples(index_array_lst)


if __name__ == '__main__':
    main()
