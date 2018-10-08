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
from util.lm_util import ler, wer, load_lm, correction, lers, wers
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
        self.df_history = pd.DataFrame(index=np.arange(num_epochs), columns=['WER', 'LER', 'ler_raw'])

        # base name for files that will be written to target directory
        self.base_name = 'model' + (f'_{self.num_minutes}_min' if self.num_minutes else '')
        print(f'base name for result files: {self.base_name}')

    def validate_epoch(self, epoch):
        K.set_learning_phase(0)

        if self.shuffle_data:
            print("shuffling validation data")
            self.data_valid.shuffle_entries()

        print(f'validating epoch {epoch+1} using best-path and beam search decoding')
        originals, results_greedy, results_beam = [], [], []
        self.data_valid.cur_index = 0  # reset index

        for _ in tqdm(range(len(self.data_valid))):
            batch_inputs, _ = next(self.data_valid)
            batch_input = batch_inputs['the_input']
            batch_input_lengths = batch_inputs['input_length']
            ground_truths = batch_inputs['source_str']

            predictions_greedy = self.decoder_greedy.decode(batch_input, batch_input_lengths)
            predictions_beam = self.decoder_beam.decode(batch_input, batch_input_lengths)

            results = self.calculate_wer_ler(ground_truths, predictions_greedy, predictions_beam)

            for result in results if self.force_output else filter(
                    lambda result: any([wer_val < 0.6 for wer_val in result['WER']]), results):
                print(tabulate(result, headers='keys', floatfmt='.4f'))

            originals = originals + ground_truths
            results_greedy = results_greedy + predictions_greedy
            results_beam = results_beam + predictions_beam

        wers_greedy, wer_mean_greedy = wers(originals, results_greedy)
        wers_beam, wer_mean_beam = wers(originals, results_beam)

        lers_greedy, ler_mean_greedy, ler_raw_greedy, ler_raw_mean_greedy = lers(originals, results_greedy)
        lers_beam, ler_mean_beam, ler_raw_beam, ler_raw_mean_beam = lers(originals, results_beam)
        print('--------------------------------------------------------')
        print(f'Validation results after epoch {epoch+1}: WER & LER using best-path and beam search decoding')
        if self.lm and self.lm_vocab:
            print(f'using LM at: {self.lm_path}')
            print(f'using LM vocab at: {self.vocab_path}')
        print('--------------------------------------------------------')
        print(f'best-path decoding:')
        print(f'  Ø WER      : {wer_mean_greedy}')
        print(f'  Ø LER      : {ler_mean_greedy}')
        print(f'  Ø LER (raw): {ler_raw_mean_greedy}')
        print(f'beam search decoding:')
        print(f'  Ø WER      : {wer_mean_beam}')
        print(f'  Ø LER      : {ler_mean_beam}')
        print(f'  Ø LER (raw): {ler_raw_mean_beam}')
        print('--------------------------------------------------------')

        self.df_history.loc[epoch] = [wer_mean_greedy, ler_mean_greedy, ler_raw_mean_greedy]

        K.set_learning_phase(1)

    def calculate_wer_ler(self, ground_truths, predictions_greedy, predictions_beam):
        results = []

        for ground_truth, pred_greedy, pred_beam in zip(ground_truths, predictions_greedy, predictions_beam):
            pred_greedy_lm = correction(pred_greedy, self.lm, self.lm_vocab)
            pred_beam_lm = correction(pred_beam, self.lm, self.lm_vocab)
            result = {
                'ground_truth': ['prediction (best path)',
                                 'prediction (best path, LM-corrected)',
                                 'prediction (beam search)',
                                 'prediction (beam search, LM-corrected)'],
                ground_truth: [pred_greedy,
                               pred_greedy_lm,
                               pred_beam,
                               pred_beam_lm],
                'LER': [ler(ground_truth, pred_greedy),
                        ler(ground_truth, pred_greedy_lm),
                        ler(ground_truth, pred_beam),
                        ler(ground_truth, pred_beam_lm)],
                'WER': [wer(ground_truth, pred_greedy),
                        wer(ground_truth, pred_greedy_lm),
                        wer(ground_truth, pred_beam),
                        wer(ground_truth, pred_beam_lm)]
            }
            results.append(result)

        return results

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
