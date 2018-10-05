import sys
from os import makedirs
from os.path import isdir, join

import keras.backend as K
import pandas as pd
from keras import callbacks
from tabulate import tabulate
from tqdm import tqdm

from core.decoder import Decoder
from text import *
from util.brnn_util import save_model


class ReportCallback(callbacks.Callback):
    def __init__(self, data_valid, model, target_dir, num_epochs, num_minutes=None, save_progress=True,
                 early_stopping=False, shuffle_data=True, force_output=False, decode_strategy='beamsearch',
                 lm_path=None, vocab_path=None):
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
        :param decode_strategy: decoder to use ('beamsearch' or 'bestpath')
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
        if lm_path:
            if not vocab_path:
                raise ValueError('ERROR: Path to vocabulary file must be supplied when supplying path to LM!')
            self.lm, self.lm_vocab = load_lm(lm_path, vocab_path)

        if not isdir(self.target_dir):
            makedirs(self.target_dir)

        self.decoder = Decoder(model, decode_strategy)

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

        print(f'validating epoch {epoch+1} using {self.decoder.decode_strategy} decoding')
        originals, results = [], []
        self.data_valid.cur_index = 0  # reset index

        validation_results = []

        for _ in tqdm(range(len(self.data_valid))):
            batch_inputs, _ = next(self.data_valid)
            batch_input = batch_inputs['the_input']
            batch_input_lengths = batch_inputs['input_length']
            ground_truths = batch_inputs['source_str']
            predictions = self.decoder.decode(batch_input, batch_input_lengths)

            for ground_truth, prediction in zip(ground_truths, predictions):
                if self.lm and self.lm_vocab:
                    using_lm = f'({self.decoder.decode_strategy} decoding, using LM)'
                    pred_lm = correction(prediction, self.lm, self.lm_vocab)
                else:
                    using_lm = f'({self.decoder.decode_strategy} decoding, not using LM)'
                    pred_lm = prediction

                ler_pred = ler(ground_truth, prediction)
                ler_lm = ler(ground_truth, pred_lm)

                wer_pred = wer(ground_truth, prediction)
                wer_lm = wer(ground_truth, pred_lm)

                if self.force_output or wer_pred < 0.4 or wer_lm < 0.4:
                    validation_results.append((ground_truth, prediction, ler_pred, wer_pred, pred_lm, ler_lm, wer_lm))

                originals.append(ground_truth)
                results.append(pred_lm)

        if validation_results:
            headers = ['Ground Truth', 'Prediction', 'LER', 'WER', 'Prediction (LM-corrected)', 'LER', 'WER']
            print(tabulate(validation_results, headers=headers, floatfmt=".4f"))
        wer_values, wer_mean = wers(originals, results)
        ler_values, ler_mean, ler_raw, ler_raw_mean = lers(originals, results)
        print('--------------------------------------------------------')
        print(f'Validation results after epoch {epoch+1}: WER & LER {using_lm}')
        print('--------------------------------------------------------')
        print(f'WER average      : {wer_mean}')
        print(f'LER average      : {ler_mean}')
        print(f'LER average (raw): {ler_raw_mean}')
        print('--------------------------------------------------------')

        self.df_history.loc[epoch] = [wer_mean, ler_mean, ler_raw_mean]

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
