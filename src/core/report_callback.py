import sys
from os import makedirs
from os.path import isdir, join

import keras.backend as K
import numpy as np
import pandas as pd
from keras import callbacks
from tabulate import tabulate
from tqdm import tqdm

from core.decoder import BeamSearchDecoder, BestPathDecoder
from util.lm_util import ler, wer, load_lm, correction, lers, wers, ler_norm
from util.rnn_util import save_model


class ReportCallback(callbacks.Callback):
    def __init__(self, data_valid, model, target_dir, num_epochs, num_minutes=None, save_progress=True,
                 early_stopping=False, shuffle_data=True, force_output=False, lm_path=None, vocab_path=None):
        """
        Will calculate WER and LER at epoch end and print out infered transcriptions from validation set using the 
        current model and weights
        :param data_valid: validation data
        :param model: compiled model
        :param target_dir: string that identifies the current run
        :param save_progress:
        :param early_stopping: 
        :param shuffle_data: 
        :param force_output:
        :param decode_strategies: list of decoding to use ('beamsearch' or 'bestpath')
        """
        super().__init__()
        self.data_valid = data_valid
        self.model = model
        self.target_dir = target_dir
        self.num_epochs = num_epochs
        self.num_minutes = num_minutes
        self.save_progress = save_progress
        self.early_stopping = early_stopping
        self.shuffle_data = shuffle_data
        self.force_output = force_output
        self.lm = None
        self.lm_vocab = None
        self.decoders = {}
        if lm_path:
            if not vocab_path:
                raise ValueError('ERROR: Path to vocabulary file must be supplied when supplying path to LM!')
            self.lm_path, self.vocab_path = lm_path, vocab_path
            self.lm, self.lm_vocab = load_lm(lm_path, vocab_path)

        if not isdir(self.target_dir):
            makedirs(self.target_dir)

        self.decoder_greedy = BestPathDecoder(model)
        self.decoder_beam = BeamSearchDecoder(model)

        # WER/LER history
        self.df_history = pd.DataFrame(index=np.arange(num_epochs),
                                       columns=['WER_greedy', 'LER_greedy', 'ler_raw_greedy',
                                                'WER_beam', 'LER_beam', 'ler_raw_beam',
                                                'WER_greedy_lm', 'LER_greedy_lm', 'ler_raw_greedy_lm',
                                                'WER_beam_lm', 'LER_beam_lm', 'ler_raw_beam_lm'
                                                ])

        # base name for files that will be written to target directory
        self.base_name = 'model' + (f'_{self.num_minutes}_min' if self.num_minutes else '')
        print(f'base name for result files: {self.base_name}')

    def validate_epoch(self, epoch):
        K.set_learning_phase(0)

        if self.shuffle_data:
            print("shuffling validation data")
            self.data_valid.shuffle_entries()

        print(f'validating epoch {epoch+1} using best-path and beam search decoding')
        originals, res_greedy, res_beam, res_greedy_lm, res_beam_lm = [], [], [], [], []
        self.data_valid.cur_index = 0  # reset index

        for _ in tqdm(range(len(self.data_valid))):
            batch_inputs, _ = next(self.data_valid)
            batch_input = batch_inputs['the_input']
            batch_input_lengths = batch_inputs['input_length']
            ground_truths = batch_inputs['source_str']

            preds_greedy = self.decoder_greedy.decode(batch_input, batch_input_lengths)
            preds_beam = self.decoder_beam.decode(batch_input, batch_input_lengths)

            preds_greedy_lm = [correction(pred_greedy, self.lm, self.lm_vocab) for pred_greedy in preds_greedy]
            preds_beam_lm = [correction(pred_beam, self.lm, self.lm_vocab) for pred_beam in preds_beam]

            results = calculate_wer_ler(ground_truths, preds_greedy, preds_beam, preds_greedy_lm, preds_beam_lm)

            for result in results if self.force_output else filter(
                    lambda result: any([wer_val < 0.6 for wer_val in result['WER']]), results):
                print()
                print(tabulate(result, headers='keys', floatfmt='.4f'))

            originals = originals + ground_truths
            res_greedy = res_greedy + preds_greedy
            res_beam = res_beam + preds_beam
            res_greedy_lm = res_greedy_lm + preds_greedy_lm
            res_beam_lm = res_beam_lm + preds_beam_lm

        wers_greedy, wer_mean_greedy = wers(originals, res_greedy)
        wers_greedy_lm, wer_mean_greedy_lm = wers(originals, res_greedy_lm)
        wers_beam, wer_mean_beam = wers(originals, res_beam)
        wers_beam_lm, wer_mean_beam_lm = wers(originals, res_beam_lm)

        lers_greedy, ler_mean_greedy, ler_raw_greedy, ler_raw_mean_greedy = lers(originals, res_greedy)
        lers_greedy_lm, ler_mean_greedy_lm, ler_raw_greedy_lm, ler_raw_mean_greedy_lm = lers(originals, res_greedy_lm)
        lers_beam, ler_mean_beam, ler_raw_beam, ler_raw_mean_beam = lers(originals, res_beam)
        lers_beam_lm, ler_mean_beam_lm, ler_raw_beam_lm, ler_raw_mean_beam_lm = lers(originals, res_beam_lm)

        table = [
            ['best-path', wer_mean_greedy, ler_mean_greedy, ler_raw_mean_greedy],
            ['beam search', wer_mean_beam, ler_mean_beam, ler_raw_mean_beam],
            ['best-path (with LM)', wer_mean_greedy_lm, ler_mean_greedy_lm, ler_raw_mean_greedy_lm],
            ['beam_search (with LM)', wer_mean_beam_lm, ler_mean_beam_lm, ler_raw_mean_beam_lm],
        ]
        headers = ['decoding strategy', 'WER', 'LER', 'LER (raw)']

        print('--------------------------------------------------------')
        print(f'Validation results after epoch {epoch+1}: WER & LER using best-path and beam search decoding')
        if self.lm and self.lm_vocab:
            print(f'using LM at: {self.lm_path}')
            print(f'using LM vocab at: {self.vocab_path}')
        print('--------------------------------------------------------')
        print(tabulate(table, headers=headers))
        print('--------------------------------------------------------')

        self.df_history.loc[epoch] = [wer_mean_greedy, ler_mean_greedy, ler_raw_mean_greedy,
                                      wer_mean_beam, ler_mean_beam, ler_raw_mean_beam,
                                      wer_mean_greedy_lm, ler_mean_greedy_lm, ler_raw_mean_greedy_lm,
                                      wer_mean_beam_lm, ler_mean_beam_lm, ler_raw_mean_beam_lm
                                      ]

        K.set_learning_phase(1)

    def finish(self):
        self.df_history.index.name = 'epoch'
        self.df_history.index += 1  # epochs start at 1
        csv_path = join(self.target_dir, f'{self.base_name}.csv')
        self.df_history.to_csv(csv_path)
        print("########################################################")
        print("Finished!")
        print(f'saved validation results to {csv_path}')
        print("########################################################")

    def on_epoch_end(self, epoch, logs=None):
        self.validate_epoch(epoch)

        if epoch == self.num_epochs - 1:
            self.finish()

        # early stopping if VAL WER worse 4 times in a row
        if self.early_stopping and self.stop_early():
            print("EARLY STOPPING")
            self.finish()

            sys.exit()

        # save checkpoint if last LER or last WER was better than all previous values
        if self.save_progress and self.new_benchmark():
            print(f'New WER or LER benchmark!')
            save_model(self.model, target_dir=self.target_dir)

    def new_benchmark(self):
        """
        We have a new benchmark if the last value in a sequence of values is the smallest
        """
        wers = self.df_history['WER'].dropna().values
        lers = self.df_history['LER'].dropna().values
        return is_last_value_smallest(wers) or is_last_value_smallest(lers)

    def stop_early(self):
        """
        stop early if last WER is bigger than all 4 previous WERs
        """
        wers = self.df_history['WER']
        if len(wers) <= 4:
            return False

        last = wers[-1]
        rest = wers[-5:-1]
        print(f'{last} vs {rest}')

        return all(val <= last for val in rest)


def is_last_value_smallest(values):
    """
    We have a new benchmark if the last value in a sequence of values is the smallest
    :param values: sequence of values
    :return:
    """
    return len(values) > 2 and values[-1] < np.min(values[:-1])


def calculate_wer_ler(ground_truths, preds_greedy, preds_beam, preds_greedy_lm, preds_beam_lm):
    results = []

    for ground_truth, pred_greedy, pred_beam, pred_greedy_lm, pred_beam_lm in zip(ground_truths,
                                                                                  preds_greedy, preds_beam,
                                                                                  preds_greedy_lm, preds_beam_lm):
        result = {
            'predictions': ['best path',
                            'beam search',
                            'best path + LM',
                            'beam search + LM'],
            ground_truth: [pred_greedy,
                           pred_beam,
                           pred_greedy_lm,
                           pred_beam_lm],
            'LER': [ler(ground_truth, pred_greedy),
                    ler(ground_truth, pred_beam),
                    ler(ground_truth, pred_greedy_lm),
                    ler(ground_truth, pred_beam_lm)],
            'LER (normalized)': [ler_norm(ground_truth, pred_greedy),
                                 ler_norm(ground_truth, pred_beam),
                                 ler_norm(ground_truth, pred_greedy_lm),
                                 ler_norm(ground_truth, pred_beam_lm)],
            'WER': [wer(ground_truth, pred_greedy),
                    wer(ground_truth, pred_beam),
                    wer(ground_truth, pred_greedy_lm),
                    wer(ground_truth, pred_beam_lm)]
        }
        results.append(result)

    return results
