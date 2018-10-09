import argparse
import re
from glob import glob
from os import listdir
from os.path import exists, abspath, join, isdir

import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    source_dir, target_dir = setup(args)
    dfs_loss, dfs_ler_wer = collect_data(source_dir)

    fig_loss, _ = plot_losses(dfs_loss)
    fig_ler_wer, _ = plot_ler_wer(dfs_ler_wer)

    fig_loss.savefig(join(target_dir, 'losses.png'))
    fig_ler_wer.savefig(join(target_dir, 'wer_ler.png'))

    plt.show()


def plot_losses(dfs_loss):
    styles = [[color + style for style in ['--', '-']] for color in ['r', 'g', 'b', 'c']]
    legends = ['training', 'validation']
    return plot_to_subplots(dfs_loss, styles, legends, 'CTC loss')


def plot_ler_wer(dfs_ler_wer):
    styles = [[color + '-' for color in ['r', 'b']]] * 4
    legends = ['LER', 'WER']
    return plot_to_subplots(dfs_ler_wer, styles, legends, 'WER/LER')


def plot_to_subplots(dfs, styles, legends, ylabel, xlabel='epochs', titles=None):
    if titles is None:
        titles = ['1 minute', '10 minutes', '100 minutes', '1,000 minutes']

    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(14, 9))

    for df, ax, title, style in zip(dfs, axes.flatten(), titles, styles):
        df.plot(ax=ax, style=style)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(legends)

    plt.tight_layout()

    return fig, axes


def setup(args):
    source_dir = abspath(args.source_dir)
    if not exists(source_dir):
        print(f'ERROR: source dir {args.source_dir} does not exist')
        exit(1)
    if not args.target_dir:
        args.target_dir = source_dir
    target_dir = abspath(args.target_dir)
    return source_dir, target_dir


def collect_data(source_dir, columns=None):
    if columns is None:
        columns = ['1_min', '10_min', '100_min', '1000_min']
    df_loss, df_val_loss = collect_tensorboard_losses(source_dir, columns)
    df_ler, df_wer = collect_ler_wer(source_dir, columns)

    dfs_loss = [pd.concat([df_loss[key], df_val_loss[key]], axis=1) for key in columns]
    dfs_ler_wer = [pd.concat([df_ler[key], df_wer[key]], axis=1) for key in columns]
    return dfs_loss, dfs_ler_wer


def collect_tensorboard_losses(source_dir, columns):
    df_loss = read_tensorflow_loss(source_dir, '-loss.csv', columns)
    df_val_loss = read_tensorflow_loss(source_dir, '-val_loss.csv', columns)
    return df_loss, df_val_loss


def read_tensorflow_loss(source_dir, postfix, columns):
    df = pd.DataFrame(columns=columns)
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', postfix).items():
        df[f'{minutes}_min'] = pd.read_csv(loss_csv)['Value'].values
    df.index += 1
    return df


def collect_ler_wer(source_dir, columns):
    df_ler = pd.DataFrame(columns=columns)
    df_wer = pd.DataFrame(columns=columns)
    for subdir_name in get_immediate_subdirectories(source_dir):
        for minutes, ler_wer_csv in map_files_to_minutes(join(source_dir, subdir_name), 'model_', '.csv').items():
            df_ler_wer = pd.read_csv(ler_wer_csv)
            df_ler[f'{minutes}_min'] = df_ler_wer['LER'].values
            df_wer[f'{minutes}_min'] = df_ler_wer['WER'].values

    df_ler.index += 1
    df_wer.index += 1
    return df_ler, df_wer


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
