import itertools
import sys
from os import makedirs
from os.path import isdir, join

import keras.backend as K
import numpy as np
import pandas as pd
from keras import callbacks

from core.decoder import BeamSearchDecoder, BestPathDecoder
from util.asr_util import calculate_metrics_mean, infer_batches_keras, lm_uses, decoding_strategies, metrics
from util.lm_util import load_lm_and_vocab
from util.log_util import print_dataframe
from util.rnn_util import save_model


class ReportCallback(callbacks.Callback):
    def __init__(self, data_valid, model, language, target_dir, num_epochs, num_minutes=None, save_progress=True,
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
        self.language = language
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
            self.lm_path, self.vocab_path = lm_path, vocab_path
            self.lm, self.lm_vocab = load_lm_and_vocab(lm_path, vocab_path)

        if not isdir(self.target_dir):
            makedirs(self.target_dir)

        self.decoder_greedy = BestPathDecoder(model, language)
        self.decoder_beam = BeamSearchDecoder(model, language)

        # WER/LER history
        columns = pd.MultiIndex.from_product([metrics, decoding_strategies, lm_uses],
                                             names=['metric', 'decoding strategy', 'LM correction'])
        self.df_history = pd.DataFrame(index=np.arange(num_epochs), columns=columns)
        # base name for files that will be written to target directory
        self.base_name = 'model' + (f'_{self.num_minutes}_min' if self.num_minutes else '')
        print(f'base name for result files: {self.base_name}')

    def validate_epoch(self, epoch):
        K.set_learning_phase(0)

        if self.shuffle_data:
            print("shuffling validation data")
            self.data_valid.shuffle_entries()

        print(f'validating epoch {epoch+1} (training on {self.num_minutes} minutes)')
        df_inferences = infer_batches_keras(self.data_valid, self.decoder_greedy, self.decoder_beam, self.language,
                                            self.lm, self.lm_vocab)

        mask = (df_inferences.loc[:, (slice(None), slice(None), 'WER')] < 0.6).any(axis=1)
        good_inferences = df_inferences[mask]

        if self.force_output or not good_inferences.empty:
            print(f'inferences with WER < 0.6 (any decoding strategy, with or without LM correction):')
            print_dataframe(good_inferences)

        mean_metrics = calculate_metrics_mean(df_inferences)

        print('--------------------------------------------------------')
        print(f'Validation results after epoch {epoch+1}: WER & LER using best-path and beam search decoding')
        if self.lm and self.lm_vocab:
            print(f'using LM at: {self.lm_path}')
            print(f'using LM vocab at: {self.vocab_path}')
        print('--------------------------------------------------------')
        print_dataframe(mean_metrics)
        print('--------------------------------------------------------')

        for m, d, l in itertools.product(metrics, decoding_strategies, lm_uses):
            self.df_history.loc[epoch][m, d, l] = mean_metrics.loc[d, l][m]

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
        We have a new benchmark if the last value in a sequence of metrics is the smallest
        """
        metrics = [
            self.df_history['WER', 'greedy', 'lm_n'].dropna().values,  # WER (best-path)
            self.df_history['WER', 'greedy', 'lm_y'].dropna().values,  # WER (best-path + LM)
            self.df_history['WER', 'beam', 'lm_n'].dropna().values,  # WER (beam search)
            self.df_history['WER', 'beam', 'lm_y'].dropna().values,  # WER (beam search + LM)
            self.df_history['LER', 'greedy', 'lm_n'].dropna().values,  # LER (best-path)
            self.df_history['LER', 'greedy', 'lm_y'].dropna().values,  # LER (best-path + LM)
            self.df_history['LER', 'beam', 'lm_n'].dropna().values,  # LER (beam search)
            self.df_history['LER', 'beam', 'lm_y'].dropna().values,  # LER (beam search + LM)
        ]
        return any(is_last_value_smallest(metric) for metric in metrics)

    def stop_early(self):
        """
        stop early if last beam search LER (before LM-correction) is bigger than all 4 previous values
        """
        lers_beam = self.df_history['LER', 'beam', 'lm_n']
        if len(lers_beam) <= 4:
            return False

        last = lers_beam[-1]
        rest = lers_beam[-5:-1]
        print(f'{last} vs {rest}')

        return all(val <= last for val in rest)


def is_last_value_smallest(values):
    """
    We have a new benchmark if the last value in a sequence of values is the smallest
    :param values: sequence of values
    :return:
    """
    return len(values) > 2 and values[-1] < np.min(values[:-1])
