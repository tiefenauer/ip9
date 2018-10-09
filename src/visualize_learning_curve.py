import argparse
import re
from glob import glob
from os import listdir
from os.path import exists, abspath, join, isdir

import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    source_dir, target_dir = setup(args)
    df = collect_data(source_dir)
    df.loc['1_min'].plot(subplots=True)
    plt.show()
    # visualize(df)


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
    df_tensorboard = collect_tensorboard_losses(source_dir)
    df_ler_wer = collect_ler_wer(source_dir)
    return pd.concat([df_tensorboard, df_ler_wer], axis=1)


def collect_tensorboard_losses(source_dir):
    df = pd.DataFrame(index=['1_min', '10_min', '100_min', '1000_min'], columns=['train_loss', 'valid_loss'])

    # collect training losses
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-loss.csv').items():
        df.loc[f'{minutes}_min', 'train_loss'] = pd.read_csv(loss_csv)['Value'].values

    # collect validation losses and merge as second column to above DataFrames
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-val_loss.csv').items():
        df.loc[f'{minutes}_min', 'valid_loss'] = pd.read_csv(loss_csv)['Value'].values

    return df


def collect_ler_wer(source_dir):
    df = pd.DataFrame(index=['1_min', '10_min', '100_min', '1000_min'], columns=['LER', 'WER'])
    for subdir_name in get_immediate_subdirectories(source_dir):
        for minutes, ler_wer_csv in map_files_to_minutes(join(source_dir, subdir_name), 'model_', '.csv').items():
            df_ler_wer = pd.read_csv(ler_wer_csv)
            df.loc[f'{minutes}_min', 'LER'] = df_ler_wer['LER'].values
            df.loc[f'{minutes}_min', 'WER'] = df_ler_wer['WER'].values

    return df


def map_files_to_minutes(root_dir, prefix, suffix):
    all_csv_files = glob(join(root_dir, '*.csv'))
    matched_files = [(re.findall(f'{prefix}(\d+)_min.*{suffix}', csv_path), csv_path) for csv_path in all_csv_files]

    return dict((int(matches[0]), csv_path) for matches, csv_path in matched_files if matches)


def get_immediate_subdirectories(root_dir):
    return [name for name in listdir(root_dir) if isdir(join(root_dir, name))]


def visualize(df):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax_wer = plot_dataframe(y=df.loc['1_min', 'WER'], metric='WER', title='WER for 1 min', ax=ax)
    plt.show()


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
