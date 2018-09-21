import argparse
import re
from glob import glob, iglob
from os import listdir
from os.path import join, abspath

import pandas as pd
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt


def visualize_learning_curve(root_dir, metric='WER'):
    print(f'visualizing learning curve from {root_dir}')
    csv_files = map_files_to_minutes(root_dir)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    for minutes, csv_path in csv_files.items():
        df = pd.read_csv(csv_path).reset_index()
        df['index'] += 1
        label = f'{minutes} minute' + ('s' if minutes > 1 else '')
        df.plot(x='index', y=metric, ax=ax, label=label)

    ax.set_xticks(range(1, len(df.index) + 1))
    ax.set_title(f'{metric} for {root_dir}')
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'{metric}')

    plt.show()


def map_files_to_minutes(root_dir):
    csv_pattern = re.compile('model_(\d*)_min\.csv')

    all_csv_files = iglob(join(root_dir, '*.csv'))
    matched_files = ((re.findall(csv_pattern, csv_path), csv_path) for csv_path in all_csv_files)

    return dict((int(matches[0]), csv_path) for matches, csv_path in matched_files if matches)


def visualize(root_dir):
    print(f'visualizing results from {root_dir}')

    csv_path = join(root_dir, 'model.csv')
    df = pd.read_csv(csv_path).reset_index()
    df['index'] += 1

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    df.plot(x='index', y='WER', ax=ax)
    df.plot(x='index', y='LER', ax=ax)

    ax.set_xticks(range(1, len(df.index) + 1))
    ax.set_title(f'WER/LER for {csv_path}')
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'WER/LER')

    plt.show()
    print('done!')


def get_latest_dir(runs_dir):
    latest_dir = sorted(listdir(runs_dir))[-1]
    return abspath(join(runs_dir, latest_dir))


def get_root_dir(runs_dir=join('..', 'runs')):
    root_dir = input('no root directory specified. Enter root directory now (or leave empty for newest)')
    return root_dir.strip() if root_dir else get_latest_dir(runs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, help='root directory containing output')
    args = parser.parse_args()

    root_dir = args.dir if args.dir else get_root_dir()
    # visualize(root_dir)
    visualize_learning_curve(root_dir, metric='LER')
