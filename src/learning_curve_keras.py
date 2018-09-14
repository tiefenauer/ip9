import argparse
import pickle
from datetime import timedelta
from os import listdir
from os.path import join, splitext

import h5py
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from tabulate import tabulate

from constants import DEFAULT_CORPUS, DEFAULT_LANGUAGE, DEFAULT_BATCH_SIZE, DEFAULT_N_EPOCHS, TARGET_ROOT
from core.callbacks import ReportCallback
from core.dataset_generator import HFS5BatchGenerator
from util.brnn_util import deep_speech_model, ctc_dummy_loss, decoder_dummy_loss, ler
from util.corpus_util import get_corpus_root
from util.log_util import redirect_to_file, create_args_str
from util.train_util import get_target_dir

parser = argparse.ArgumentParser(description="""Train a simplified DeepSpeech model""")
parser.add_argument('-c', '--corpus', type=str, choices=['rl', 'ls'], nargs='?', default=DEFAULT_CORPUS,
                    help=f'(optional) corpus on which to train (rl=ReadyLingua, ls=LibriSpeech). Default: {DEFAULT_CORPUS}')
parser.add_argument('-l', '--language', type=str, nargs='?', default=DEFAULT_LANGUAGE,
                    help=f'(optional) language on which to train the RNN. Default: {DEFAULT_LANGUAGE}')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=DEFAULT_BATCH_SIZE,
                    help=f'(optional) number of speech segments to include in one batch (default: {DEFAULT_BATCH_SIZE})')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TARGET_ROOT,
                    help=f'(optional) root of folder where results will be written to (default: {TARGET_ROOT})')
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=DEFAULT_N_EPOCHS,
                    help=f'(optional) number of epochs to train the model (default: {DEFAULT_N_EPOCHS})')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    target_dir = get_target_dir('DeepSpeech', args)
    log_file_path = join(target_dir, 'train.log')
    redirect_to_file(log_file_path)
    print(f'All results and artifacts will be written to: {target_dir}')

    for num_minutes in [1, 10, 100, 1000]:
        # -------------------------------------------------------------
        # some Keras/TF setup
        # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "2"
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)
        # -------------------------------------------------------------

        print(f'training on {num_minutes} minutes of audio data ({timedelta(minutes=num_minutes)})')
        model, history = train_model(target_dir, args.corpus, args.language, args.batch_size, num_minutes)

        model_path = join(target_dir, f'model_{num_minutes}_minutes.h5')
        history_path = join(target_dir, f'history_{num_minutes}_minutes.pkl')
        print(f'saving model to {model_path} and history to {history_path}')
        with open(history_path, 'wb') as history_f:
            model.save(model_path)
            pickle.dump(history.history, history_f)

        K.clear_session()


def train_model(target_dir, corpus_id, language, batch_size, num_minutes):
    train_it, dev_it, test_it = generate_subsets(corpus_id, language, batch_size, num_minutes=num_minutes)

    model = deep_speech_model(num_features=13)
    model.summary()

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(loss={'ctc': ctc_dummy_loss, 'decoder': decoder_dummy_loss},
                  optimizer=opt,
                  metrics={'decoder': ler},  #
                  loss_weights=[1, 0]  # only optimize CTC cost
                  )

    cb_list = []
    tb_cb = TensorBoard(log_dir=target_dir, write_graph=True, write_images=True)
    cb_list.append(tb_cb)

    report_cb = ReportCallback(model, dev_it, target_dir)
    cb_list.append(report_cb)

    history = model.fit_generator(generator=train_it,
                                  validation_data=dev_it,
                                  epochs=args.num_epochs,
                                  callbacks=cb_list,
                                  verbose=1
                                  )

    return model, history


def generate_subsets(corpus_id, language, batch_size, feature_type='mfcc', num_minutes=None):
    corpus_root = get_corpus_root(corpus_id)
    feature_file = find_precomputed_features(corpus_root, feature_type)
    if feature_file:
        print(f'found precomputed features: {feature_file}. Using HDF5-Features')
        f = h5py.File(feature_file, 'r')
        train_it = HFS5BatchGenerator(f['train'][language], feature_type, batch_size, num_minutes=num_minutes)
        dev_it = HFS5BatchGenerator(f['dev'][language], feature_type, batch_size)
        test_it = HFS5BatchGenerator(f['test'][language], feature_type, batch_size)

        print()
        table = [
            ['', 'Train-Set', 'Dev-Set', 'Test-Set'],
            ['# speech segments', train_it.n, dev_it.n, test_it.n],
            [f'# batches (batch_size={batch_size})', len(train_it), len(dev_it), len(test_it)],
            ['total audio length'] + [timedelta(seconds=subset.audio_total_length) for subset in [train_it, dev_it, test_it]]

        ]
        print(tabulate(table, headers='firstrow'))
        print()
        return train_it, dev_it, test_it

    raise ValueError(f'no precomputed HDF5-features found in {corpus_root}!')


def find_precomputed_features(corpus_root, feature_type):
    h5_features = list(join(corpus_root, file) for file in listdir(corpus_root)
                       if splitext(file)[0].startswith('features')
                       and feature_type in splitext(file)[0]
                       and splitext(file)[1] == '.h5')
    default_features = f'features_{feature_type}.h5'
    return default_features if default_features in h5_features else h5_features[0] if h5_features else None


if __name__ == '__main__':
    main()
