import argparse
from os import listdir
from os.path import join, abspath

import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def visualize(root_dir):
    print(f'visualizing results from {root_dir}')
    csv_path = join(root_dir, 'wer_ler.csv')
    df = pd.read_csv(csv_path).reset_index()
    df['index'] += 1

    ax = df.plot(x='index', y=['WER', 'LER'], figsize=(16, 9), xticks=range(1, len(df.index) + 1))
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
    visualize(root_dir)
