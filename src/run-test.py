#!/usr/bin/env python
import argparse
from datetime import datetime
from os.path import abspath

from keras.optimizers import SGD

from core.batch_generator import CSVBatchGenerator
from core.report_callback import *
from util.lm_util import load_lm_and_vocab
from util.log_util import create_args_str
from util.rnn_util import load_keras_model, create_keras_session

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=False,
                    help='path to trained Keras model')
parser.add_argument('-t', '--target_dir', type=str, required=False,
                    help='target directory for results (optional). If not set, results will be written to model_dir')
parser.add_argument('-f', '--test_files', type=str, default='',
                    help='list of all test files, seperate by a comma if multiple')
parser.add_argument('-b', '--test_batches', type=int, default=0,
                    help='number of batches to use for testing (default: all)')
parser.add_argument('-s', '--batch_size', type=int, default=16,
                    help='batch size to use for testing (default: 16)')
parser.add_argument('-l', '--lm', type=str, default='',
                    help='path to compiled KenLM binary for spelling correction (optional)')
parser.add_argument('-a', '--lm_vocab', type=str, default='',
                    help='(optional) path to text file containing vocabulary used to train KenLM model. The vocabulary '
                         'must be words separated by a single whitespace without newlines. A vocabulary is mandatory '
                         'if a LM is supplied with \'--lm_path\'. If \'--lm_path\' is set and  lm_vocab_path not, a '
                         'default vocabulary file with the same name as lm_path and the ending \'.vocab\' '
                         'will be searched. If this is not found, the script will exit.')
parser.add_argument('--language', type=str, choices=['en', 'de'], default='en',
                    help='language to train on. '
                         'English will use 26 characters from the alphabet, German 29 (umlauts)')
parser.add_argument('-g', '--gpu', type=str, default=None, required=False,
                    help='GPU to use (optional). If not set, you will be asked at runtime')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    target_dir = setup(args)
    print()
    print(f'all output will be written to {target_dir}')
    print()

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = load_keras_model(args.model_dir, opt)
    model.summary()

    lm, vocab = None, None
    if args.lm:
        lm, vocab = load_lm_and_vocab(args.lm, args.lm_vocab)

    test_model(model, args.test_files, args.test_batches, args.batch_size, args.language, lm, vocab, target_dir)


def setup(args):
    if not args.model_dir:
        args.model_dir = input('Enter directory where H5 file ist stored: ')
    if not args.target_dir:
        args.target_dir = args.model_dir

    target_dir = abspath(args.target_dir)
    if not isdir(target_dir):
        makedirs(target_dir)

    create_keras_session(args.gpu)
    return target_dir


def test_model(model, test_files, test_batches, batch_size, language, lm, lm_vocab, target_dir):
    data_test = CSVBatchGenerator(test_files, language, sort=False, n_batches=test_batches, batch_size=batch_size)

    print(f'Testing model with samples from {test_files}')
    decoder_greedy = BestPathDecoder(model, language)
    decoder_beam = BeamSearchDecoder(model, language)
    df_inferences = infer_batches_keras(data_test, decoder_greedy, decoder_beam, language, lm, lm_vocab)
    df_metrics_means = calculate_metrics_mean(df_inferences)

    csv_path = join(target_dir, 'test_report.csv')
    txt_path = join(target_dir, 'test_report.txt')

    with open(txt_path, 'w') as f:
        print_dataframe(df_inferences)
        print_dataframe(df_metrics_means)

        df_inferences.to_csv(csv_path)
        f.write(f"""
        Report from {datetime.now()}\n
        inferred samples from {test_files} ({test_batches} batches)\n
        inferred transcript saved to {csv_path}\n
        mean metrics:\n\n
        {str(df_metrics_means)})""")

    print(f'Testing done! Resutls saved to {csv_path}. LER/WER stats saved to {txt_path}')


if __name__ == '__main__':
    main()
