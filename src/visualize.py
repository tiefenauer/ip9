import argparse
import re
from glob import iglob
from os import listdir
from os.path import join, abspath

import pandas as pd
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt


def visualize_single_run(csv_path):
    print(f'visualizing results from {csv_path}')

    df = pd.read_csv(csv_path).reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    plot_dataframe(df, 'WER', csv_path, ax=ax)
    plot_dataframe(df, 'LER', csv_path, ax=ax)

    plt.show()


def visualize_learning_curves(root_dir):
    print(f'visualizing learning curve from {root_dir}')
    csv_files = map_files_to_minutes(root_dir)

    fig, (ax_wer, ax_ler) = plt.subplots(2, 1, figsize=(16, 16))

    for minutes, csv_path in csv_files.items():
        df = pd.read_csv(csv_path).reset_index()
        label = f'{minutes} minute' + ('s' if minutes > 1 else '')
        ax_wer = plot_dataframe(df, 'WER', root_dir, ax=ax_wer, label=label)
        ax_ler = plot_dataframe(df, 'LER', root_dir, ax=ax_ler, label=label)

    plt.show()


def map_files_to_minutes(root_dir):
    csv_pattern = re.compile('model_(\d*)_min\.csv')

    all_csv_files = iglob(join(root_dir, '*.csv'))
    matched_files = ((re.findall(csv_pattern, csv_path), csv_path) for csv_path in all_csv_files)

    return dict((int(matches[0]), csv_path) for matches, csv_path in matched_files if matches)


def plot_dataframe(df, metric, data_source, ax=None, x='index', label=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    if not label:
        label = metric

    df['index'] += 1
    df.plot(x=x, y=metric, ax=ax, label=label)

    ax.set_xticks(range(1, len(df.index) + 1))
    ax.set_title(f'{metric} for {data_source}')
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'{metric}')
    return ax


def get_latest_dir(runs_dir):
    latest_dir = sorted(listdir(runs_dir))[-1]
    return abspath(join(runs_dir, latest_dir))


def get_root_dir(runs_dir=join('..', 'runs')):
    root_dir = input('no root directory specified. Enter root directory now (or leave empty for newest): ')
    return root_dir.strip() if root_dir else get_latest_dir(runs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, help='root directory containing output')
    args = parser.parse_args()

    root_dir = args.dir if args.dir else get_root_dir()

    # model.csv contains the results of a single training run
    if 'model.csv' in listdir(root_dir):
        visualize_single_run(join(root_dir, 'model.csv'))

    # model_[0-9]*_min.csv files contain the results of several training runs to plot a learning curve
    else:
        visualize_learning_curves(root_dir)
