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
    df_loss, df_val_loss, df_ler, df_ler_wer = collect_data(source_dir)
    fig_loss, _ = plot_losses(df_loss, df_val_loss)
    plot_ler_wer(df_ler_wer)
    plt.show()
    fig_loss.savefig(join(target_dir, 'losses.png'))


def plot_losses(df_loss, df_val_loss):
    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, constrained_layout=True, figsize=(14, 9))

    colors = ['r', 'g', 'b', 'c']
    for i, (key, color) in enumerate(zip(df_loss.keys(), colors)):
        ax = axes.flatten()[i]
        loss = df_loss[key]
        val_loss = df_val_loss[key]
        df = pd.concat([loss, val_loss], axis=1)
        df.columns = ['train_loss', 'val_loss']
        df.plot(ax=ax, style=[color + style for style in ['--', '-']])

        ax.set_title(create_title(key))
        ax.set_xlabel('epochs')
        ax.set_ylabel('CTC loss')
        ax.legend(['training', 'validation'])

    plt.tight_layout()

    return fig, axes


def create_title(key):
    pattern = re.compile('(\d*)_min')
    minutes = int(re.findall(pattern, key)[0])
    unit = 'minutes' if minutes > 1 else 'minute'
    return f'{minutes:,} {unit}'


def plot_ler_wer(df):
    ax = df.plot()
    ax.set_title('LER and WER')
    ax.set_xlabel('epochs')
    ax.set_ylabel('LER/WER')


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
    df_loss, df_val_loss = collect_tensorboard_losses(source_dir)
    df_ler, df_wer = collect_ler_wer(source_dir)
    df_loss.index += 1
    df_loss.index.name = 'index'
    df_val_loss.index += 1
    df_ler.index += 1
    df_wer.index += 1
    return df_loss, df_val_loss, df_ler, df_ler


def collect_tensorboard_losses(source_dir):
    df_loss = pd.DataFrame(columns=['1_min', '10_min', '100_min', '1000_min'])
    df_val_loss = pd.DataFrame(columns=['1_min', '10_min', '100_min', '1000_min'])

    # collect training losses
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-loss.csv').items():
        df_loss[f'{minutes}_min'] = pd.read_csv(loss_csv)['Value'].values

    # collect validation losses and merge as second column to above DataFrames
    for minutes, loss_csv in map_files_to_minutes(source_dir, 'run_', '-val_loss.csv').items():
        df_val_loss[f'{minutes}_min'] = pd.read_csv(loss_csv)['Value'].values

    return df_loss, df_val_loss


def collect_ler_wer(source_dir):
    df_ler = pd.DataFrame(columns=['1_min', '10_min', '100_min', '1000_min'])
    df_wer = pd.DataFrame(columns=['1_min', '10_min', '100_min', '1000_min'])
    for subdir_name in get_immediate_subdirectories(source_dir):
        for minutes, ler_wer_csv in map_files_to_minutes(join(source_dir, subdir_name), 'model_', '.csv').items():
            df_ler_wer = pd.read_csv(ler_wer_csv)
            df_ler[f'{minutes}_min'] = df_ler_wer['LER'].values
            df_wer[f'{minutes}_min'] = df_ler_wer['WER'].values

    return df_ler, df_wer


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
