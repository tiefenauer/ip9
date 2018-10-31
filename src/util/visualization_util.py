import itertools
import re
from genericpath import isdir
from glob import glob
from os import listdir
from os.path import dirname, join, relpath
from pathlib import Path

import pandas as pd
import seaborn as sns
import tensorflow as tf

from util.asr_util import metrics, decoding_strategies, lm_uses

sns.set()
from matplotlib import pyplot as plt


def visualize_learning_curve(source_dir, silent=False):
    """
    Visualize the training progress by plotting learning curves for
    - the training- and validation-loss
    - the metrics (LER and WER)

    The plots will be saved as PNG in the source directory.

    :param source_dir: path to the directory containing the data of the training runs
    :param silent: whether to show the plots
    """
    df_losses = collect_loss(source_dir)
    df_metrics = collect_metrics(source_dir)
    fig_loss, _ = plot_loss(df_losses)
    fig_wer, *_ = plot_metric(df_metrics, 'wer')
    fig_ler, *_ = plot_metric(df_metrics, 'ler')
    fig_loss.savefig(join(source_dir, 'loss.png'))
    fig_wer.savefig(join(source_dir, 'wer.png'))
    fig_ler.savefig(join(source_dir, 'ler.png'))
    if not silent:
        plt.show()


def collect_loss(source_dir):
    data = {}
    for tensorboard_file in sorted(glob(join(source_dir, '**/tensorboard/event*'))):
        print(f'parsing {tensorboard_file}')
        loss, val_loss = [], []
        minutes = Path(relpath(tensorboard_file, source_dir)).parts[0]

        for e in tf.train.summary_iterator(tensorboard_file):
            loss += [v.simple_value for v in e.summary.value if v.tag == 'loss']
            val_loss += [v.simple_value for v in e.summary.value if v.tag == 'val_loss']
        data[('CTC_train', minutes)] = loss
        data[('CTC_val', minutes)] = val_loss

    df = pd.DataFrame(data)
    df.index.name = 'epoch'
    df.index += 1
    return df


def collect_metrics(source_dir):
    columns = pd.MultiIndex.from_product([
        metrics,
        decoding_strategies,
        lm_uses,
        ['1_min', '10_min', '100_min', '1000_min']
    ])
    df_metric = pd.DataFrame(columns=columns)

    for subdir_name in [name for name in listdir(source_dir) if isdir(join(source_dir, name))]:
        minutes = re.findall('(\d*)', subdir_name)[0]
        ler_wer_csv = join(source_dir, subdir_name, f'model_{minutes}_min.csv')

        df = pd.read_csv(ler_wer_csv, header=[0, 1, 2], index_col=0)
        for m, d, l in itertools.product(metrics, decoding_strategies, lm_uses):
            df_metric[m, d, l, f'{minutes}_min'] = df[m, d, l]
    df_metric.index += 1
    return df_metric


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
    ax.set_xlim(1)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    ax.set_title(title)


def map_files_to_minutes(root_dir, prefix, suffix):
    all_csv_files = glob(join(root_dir, '*.csv'))
    matched_files = [(re.findall(f'{prefix}(\d+)_min.*{suffix}', csv_path), csv_path) for csv_path in all_csv_files]

    return dict((int(matches[0]), csv_path) for matches, csv_path in matched_files if matches)


def visualize_pipeline_performance(csv_path, silent=False):
    """
    Visualize the performance of a pipeline by creating scatter plots for various correlations from a CSV file:
    - transcript length vs. LER
    - transcript length vs. Levenshtein similarity
    - LER vs. Levenshtein similarity

    The plots will be saved as PNG in the same directory as the CSV

    :param csv_path: path to CSV file containing the data
    :param silent: whether to show plots at the end
    """
    target_dir = dirname(csv_path)
    fig, (ax_ler, ax_similarity, ax_correlation) = plt.subplots(1, 3, figsize=(14, 5))

    df = pd.read_csv(csv_path)

    colors = 'bg'
    parms = {}
    for i, engine_id in enumerate(df.engine_id.unique()):
        parms[engine_id] = {}
        parms[engine_id]['color'] = colors[i]
        parms[engine_id]['label'] = engine_id

    for (engine_id, engine_path), df_engine in df.groupby(['engine_id', 'engine_path']):
        color = parms[engine_id]['color']
        label = parms[engine_id]['label']

        df_engine.plot.scatter(x='transcript_length', y='LER', c=color, label=label, ax=ax_ler)
        df_engine.plot.scatter(x='transcript_length', y='similarity', c=color, label=label, ax=ax_similarity)
        df_engine.plot.scatter(x='LER', y='similarity', c=color, label=label, ax=ax_correlation)

    fig.suptitle('Comparing transcript length against WER and Levenshtein similarity')
    ax_ler.set_title('LER (lower is better)')
    ax_similarity.set_title('Levenshtein similarity (higher is better)')
    ax_correlation.set_title('Correlation between LER and similarity')

    ax_ler.set_xlabel('transcript length (characters)')
    ax_similarity.set_xlabel('transcript length (characters)')
    # fig.tight_layout()
    fig.savefig(join(target_dir, f'performance.png'))
    if not silent:
        plt.show()
