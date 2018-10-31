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

parser = argparse.ArgumentParser()
parser.add_argument('source_dir', type=str, help='root directory containing output', nargs='?')
parser.add_argument('-s', '--silent', action='store_true', help='(optional) whether to suppress showing plots')
args = parser.parse_args()


def main(args):
    source_dir = setup(args)
    df_losses = collect_loss(source_dir)
    df_metrics = collect_metrics(source_dir)

    fig_loss, _ = plot_loss(df_losses)
    fig_wer, *_ = plot_metric(df_metrics, 'wer')
    fig_ler, *_ = plot_metric(df_metrics, 'ler')

    fig_loss.savefig(join(source_dir, 'loss.png'))
    fig_wer.savefig(join(source_dir, 'wer.png'))
    fig_ler.savefig(join(source_dir, 'ler.png'))

    if not args.silent:
        plt.show()


def plot_loss(df_losses):
    fig, ax = plt.subplots(figsize=(14, 7))

    legend = [f'{mins} ({train_valid})' for train_valid in ['training', 'validation'] for mins in
              ['1 min', '10 mins', '100 mins', '1000 mins']]
    styles = [color + line for (line, color) in itertools.product(['-', '--'], list('rgbm'))]

    plot_df(df_losses, ax=ax, styles=styles, legend=legend, ylabel='loss', title='CTC-Loss')

    fig.tight_layout()
    return fig, ax


def plot_metric(df_metrics, metric):
    metric = metric.upper()
    df_greedy = df_metrics[metric, 'greedy']
    df_beam = df_metrics[metric, 'beam']

    fig, (ax_greedy, ax_beam) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    legend = [f'{mins}{lm_use}' for lm_use in ['', '+LM'] for mins in ['1 min', '10 mins', '100 mins', '1000 mins']]
    styles = [color + line for (line, color) in itertools.product(['-', '--'], list('rgbm'))]

    plot_df(df_greedy, ax=ax_greedy, styles=styles, legend=legend, ylabel=metric, title='best-path decoding')
    plot_df(df_beam, ax=ax_beam, styles=styles, legend=legend, ylabel=metric, title='beam search decoding')

    fig.tight_layout()
    fig.suptitle(metric)
    return fig, ax_greedy, ax_beam


def plot_df(df, ax, styles, legend, ylabel, title):
    df.plot(style=styles, ax=ax)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    ax.set_title(title)


def plot_to_subplots(df, axes, styles):
    for key, ax, style in zip(['1_min', '10_min', '100_min', '1000_min'], axes.flatten(), styles):
        df[key].plot(ax=ax, style=style)
        ax.set_title(key.replace('_', ' '))


def setup(args):
    if not args.source_dir:
        args.source_dir = input('Enter source directory: ')
    source_dir = abspath(args.source_dir)
    if not exists(source_dir):
        print(f'ERROR: source dir {args.source_dir} does not exist')
        exit(1)
    return source_dir


def collect_loss(source_dir):
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
    df_metric = pd.DataFrame(columns=columns)

    for subdir_name in get_immediate_subdirectories(source_dir):
        for minutes, ler_wer_csv in map_files_to_minutes(join(source_dir, subdir_name), 'model_', '.csv').items():
            df = pd.read_csv(ler_wer_csv, header=[0, 1, 2], index_col=0)
            for m, d, l in itertools.product(metrics, decoding_strategies, lm_uses):
                df_metric[m, d, l, f'{minutes}_min'] = df[m, d, l]
    df_metric.index += 1
    return df_metric


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
    main(args)
