#!/usr/bin/env python
import argparse
from datetime import datetime
from os.path import abspath

from keras.optimizers import SGD

from core.batch_generator import CSVBatchGenerator
from core.report_callback import *
from util.log_util import create_args_str
from util.rnn_util import load_model_from_dir, create_keras_session

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=True,
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
                    help='path to vocabulary of LM (mandatory, if lm is set!)')
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
    model = load_model_from_dir(args.model_dir, opt)
    model.summary()

    lm, vocab = None, None
    if args.lm and args.lm_vocab:
        lm, vocab = load_lm(args.lm, args.lm_vocab)

    test_model(model, args.test_files, args.test_batches, args.batch_size, lm, vocab, target_dir)


def setup(args):
    if not args.target_dir:
        args.target_dir = args.model_dir

    target_dir = abspath(args.target_dir)
    if not isdir(target_dir):
        makedirs(target_dir)

    create_keras_session(args.gpu)
    return target_dir


def test_model(model, test_files, test_batches, batch_size, lm, lm_vocab, target_dir):
    data_test = CSVBatchGenerator(test_files, sort=False, n_batches=test_batches, batch_size=batch_size)
    decoder_greedy = BestPathDecoder(model)
    decoder_beam = BeamSearchDecoder(model)

    print(f'Testing model with samples from {test_files} using {decoder_greedy.strategy} decoding')
    results = []
    for _ in tqdm(range(len(data_test)), unit=' batches'):
        batch_inputs, _ = next(data_test)
        ground_truths, preds_greedy, preds_greedy_lm, preds_beam, preds_beam_lm = infer_batch(batch_inputs,
                                                                                              decoder_greedy,
                                                                                              decoder_beam,
                                                                                              lm, lm_vocab)
        results += calculate_wer_ler(ground_truths, preds_greedy, preds_beam, preds_greedy_lm, preds_beam_lm)

    df_results = pd.concat(results).sort_values(by='LER')

    df_means = pd.DataFrame(index=pd.MultiIndex.from_product([['Ø LER', 'Ø WER'], ['best-path', 'beam search']]),
                            columns=['without LM', 'with LM'])
    df_means.loc['Ø LER', 'best-path']['without LM'] = df_results.loc['best-path', 'lm_n']['LER'].mean()
    df_means.loc['Ø LER', 'best-path']['with LM'] = df_results.loc['beam search', 'lm_y']['LER'].mean()
    df_means.loc['Ø LER', 'beam search']['without LM'] = df_results.loc['best-path', 'lm_n']['LER'].mean()
    df_means.loc['Ø LER', 'beam search']['with LM'] = df_results.loc['beam search', 'lm_y']['LER'].mean()
    df_means.loc['Ø WER', 'best-path']['without LM'] = df_results.loc['best-path', 'lm_n']['WER'].mean()
    df_means.loc['Ø WER', 'best-path']['with LM'] = df_results.loc['beam search', 'lm_y']['WER'].mean()
    df_means.loc['Ø WER', 'beam search']['without LM'] = df_results.loc['best-path', 'lm_n']['WER'].mean()
    df_means.loc['Ø WER', 'beam search']['with LM'] = df_results.loc['beam search', 'lm_y']['WER'].mean()

    csv_path = join(target_dir, 'test_report.csv')
    txt_path = join(target_dir, 'test_report.txt')

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.max_colwidth', -1), \
         open(txt_path, 'w') as f:
        print(df_results)
        print(df_means)

        df_results.to_csv(csv_path)
        f.write(f"""
        Report from {datetime.now()}\n
        inferred samples from {test_files} ({test_batches} batches)\n
        inferred transcript saved to {csv_path}\n
        mean metrics:\n\n
        {str(df_means)})""")

    print(f'Testing done! Resutls saved to {csv_path}. LER/WER stats saved to {txt_path}')


if __name__ == '__main__':
    main()
