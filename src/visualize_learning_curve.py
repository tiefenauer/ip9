import argparse
import itertools
import re
from glob import glob
from os import listdir
from os.path import exists, abspath, join, isdir

import seaborn as sns

from util.asr_util import metrics, decoding_strategies, lm_uses

sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    source_dir, target_dir = setup(args)
    df_losses, df_metrics = collect_data(source_dir)

    fig_loss, _ = plot_losses(df_losses)
    fig_best, fig_beam, _, _ = plot_metrics(df_metrics)

    fig_loss.savefig(join(target_dir, 'losses.png'))
    fig_best.savefig(join(target_dir, 'wer_ler_best.png'))
    fig_beam.savefig(join(target_dir, 'wer_ler_beam.png'))

    plt.show()


def plot_losses(df_losses):
    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(14, 9))
    plot_to_subplots(df_losses['CTC_train'], axes, ['r--'] * 4)
    plot_to_subplots(df_losses['CTC_val'], axes, ['r-'] * 4)

    [ax.legend(['training', 'validation']) for ax in axes.flatten()]
    [ax.set_xlabel('epochs') for ax in axes.flatten()]
    [ax.set_ylabel('CTC loss') for ax in axes.flatten()]

    fig.tight_layout()
    return fig, axes


def plot_metrics(df_metrics):
    fig_best, axes_best = plot_metric(df_metrics, 'greedy')
    fig_beam, axes_beam = plot_metric(df_metrics, 'beam')
    return fig_best, fig_beam, axes_best, axes_beam


def plot_metric(df_metrics, decoding_strategy):
    df_wer = df_metrics['WER', decoding_strategy, 'lm_n']
    df_wer_lm = df_metrics['WER', decoding_strategy, 'lm_y']
    df_ler = df_metrics['LER', decoding_strategy, 'lm_n']
    df_ler_lm = df_metrics['LER', decoding_strategy, 'lm_y']

    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(14, 9))

    plot_to_subplots(df_wer, axes, ['r--'] * 4)
    plot_to_subplots(df_wer_lm, axes, ['r-'] * 4)
    plot_to_subplots(df_ler, axes, ['b--'] * 4)
    plot_to_subplots(df_ler_lm, axes, ['b-'] * 4)

    [ax.legend(['WER', ' WER + LM', 'LER', ' LER + LM']) for ax in axes.flatten()]
    [ax.set_xlabel('epoch') for ax in axes.flatten()]
    [ax.set_ylabel('WER/LER') for ax in axes.flatten()]

    fig.tight_layout()
    return fig, axes


def plot_to_subplots(df, axes, styles):
    for key, ax, style in zip(['1_min', '10_min', '100_min', '1000_min'], axes.flatten(), styles):
        df[key].plot(ax=ax, style=style)
        ax.set_title(key.replace('_', ' '))


def setup(args):
    source_dir = abspath(args.source_dir)
    if not exists(source_dir):
        print(f'ERROR: source dir {args.source_dir} does not exist')
        exit(1)
    if not args.target_dir:
        args.target_dir = source_dir
    target_dir = abspath(args.target_dir)
    return source_dir, target_dir


def collect_data(source_dir):
    df_losses = collect_losses(source_dir)
    df_metrics = collect_metrics(source_dir)
    return df_losses, df_metrics


def collect_losses(source_dir):
    columns = [
        ['CTC_train', 'CTC_val'],
        ['1_min', '10_min', '100_min', '1000_min']
    ]

    df_losses = pd.DataFrame(columns=pd.MultiIndex.from_product(columns))

    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-loss.csv').items():
        df_losses['CTC_train', f'{minutes}_min'] = pd.read_csv(loss_csv)['Value']
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-val_loss.csv').items():
        df_losses['CTC_val', f'{minutes}_min'] = pd.read_csv(loss_csv)['Value']

    df_losses.index += 1
    return df_losses


def collect_metrics(source_dir):
    columns = pd.MultiIndex.from_product([
        metrics,
        decoding_strategies,
        lm_uses,
        ['1_min', '10_min', '100_min', '1000_min']
    ])

    df_ler_wer = pd.DataFrame(columns=columns)
    for subdir_name in get_immediate_subdirectories(source_dir):
        for minutes, ler_wer_csv in map_files_to_minutes(join(source_dir, subdir_name), 'model_', '.csv').items():
            df = pd.read_csv(ler_wer_csv, header=[0, 1, 2], index_col=0)
            for m, d, l in itertools.product(metrics, decoding_strategies, lm_uses):
                df_ler_wer[m, d, l, f'{minutes}_min'] = df[m, d, l]
    df_ler_wer.index += 1
    return df_ler_wer


def map_files_to_minutes(root_dir, prefix, suffix):
    all_csv_files = glob(join(root_dir, '*.csv'))
    matched_files = [(re.findall(f'{prefix}(\d+)_min.*{suffix}', csv_path), csv_path) for csv_path in all_csv_files]

    return dict((int(matches[0]), csv_path) for matches, csv_path in matched_files if matches)


def get_immediate_subdirectories(root_dir):
    return [name for name in listdir(root_dir) if isdir(join(root_dir, name))]


def plot_dataframe(y, metric, title, ax=None, label=None):
    if not label:
        label = metric

    x = range(1, len(y) + 1)
    ax.plot(x=x, y=y, label=label)

    ax.set_xticks(x)
    ax.set_title(title)
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'{metric}')
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True,
                        help='root directory containing output')
    parser.add_argument('--target_dir', nargs='?', type=str,
                        help='(optional) target directory to save plots. Root directory will be used if not set.')
    args = parser.parse_args()
    main(args)
