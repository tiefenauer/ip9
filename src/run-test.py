#!/usr/bin/env python
import argparse
import datetime
from operator import itemgetter

from keras.optimizers import SGD

from core.batch_generator import CSVBatchGenerator
from core.report_callback import *
from util.log_util import create_args_str
from util.rnn_util import load_model

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=True,
                    help='path to trained Keras model')
parser.add_argument('-t', '--target_dir', type=str, required=True,
                    help='target directory for results (required)')
parser.add_argument('-r', '--run_id', type=str, required=False,
                    help='ID of test run (optional)')
parser.add_argument('-f', '--test_files', type=str, default='',
                    help='list of all test files, seperate by a comma if multiple')
parser.add_argument('-b', '--test_batches', type=int, default=0,
                    help='number of batches to use for testing (default: all)')
parser.add_argument('-s', '--batch_size', type=int, default=16,
                    help='batch size to use for testing (default: 16)')
parser.add_argument('-x', '--decoder', type=str, default='beamsearch',
                    help='decoder to use. Valid values are: old | bestpath | beamsearch (default: beamsearch)')
parser.add_argument('-l', '--lm', type=str, default='',
                    help='path to compiled KenLM binary for spelling correction (optional)')
parser.add_argument('-v', '--lm_vocab', type=str, default='',
                    help='path to vocabulary of LM (mandatory, if lm is set!)')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    target_dir = setup(args)
    print()
    print(f'all output will be written to {target_dir}')
    print()

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = load_model(args.model_dir, opt)
    model.summary()

    lm, vocab = None, None
    if args.lm and args.lm_vocab:
        lm, vocab = load_lm(args.lm, args.lm_vocab)

    test_model(model, args.test_files, args.test_batches, args.batch_size, args.decoder, lm, vocab, target_dir)


def setup(args):
    if not args.run_id:
        args.run_id = 'DS_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    target_dir = join(args.target_dir, args.run_id)
    if not isdir(target_dir):
        makedirs(target_dir)

    return target_dir


def test_model(model, test_files, test_batches, batch_size, decode_strategy, lm, vocab, target_dir):
    data_test = CSVBatchGenerator(test_files, sort=False, n_batches=test_batches, batch_size=batch_size)
    decoder = Decoder(model, decode_strategy)
    test_results = []

    print(f'Testing model with samples from {test_files} using {decode_strategy} decoding')
    for _ in tqdm(range(len(data_test)), unit=' batches'):
        batch_inputs, _ = next(data_test)
        batch_input = batch_inputs['the_input']
        batch_input_lengths = batch_inputs['input_length']
        ground_truths = batch_inputs['source_str']
        predictions = decoder.decode(batch_input, batch_input_lengths)

        for ground_truth, prediction in zip(ground_truths, predictions):
            pred_lm = predictions
            if lm and vocab:
                pred_lm = correction(prediction, lm, vocab)

            ler_pred = ler(ground_truth, prediction)
            ler_lm = ler(ground_truth, pred_lm)
            wer_pred = wer(ground_truth, prediction)
            wer_lm = wer(ground_truth, pred_lm)
            test_results.append((ground_truth, prediction, ler_pred, wer_pred, pred_lm, ler_lm, wer_lm))

    # sort by WER (LM-corrected) and calculate metrics
    test_results = sorted(test_results, key=itemgetter(6))
    headers = ['Ground Truth', 'Prediction', 'LER', 'WER', 'Prediction (LM)', 'LER (with LM)', 'WER (with LM)']
    df = pd.DataFrame(test_results, columns=headers)

    print()
    stats = tabulate(
        headers=['without LM', 'with LM'],
        tabular_data=[
            ['Ø LER', df['LER'].mean(), df['LER (with LM)'].mean()],
            ['Ø WER', df['WER'].mean(), df['WER (with LM)'].mean()]],
        floatfmt=".4f")
    print(stats)
    print()

    csv_path = join(target_dir, 'test_results.csv')
    df.to_csv(csv_path)
    txt_path = join(target_dir, 'test_results.txt')
    with open(txt_path, 'w') as f:
        f.write(stats)
    print(f'Testing done! Resutls saved to {csv_path}. LER/WER stats saved to {txt_path}')


if __name__ == '__main__':
    main()
