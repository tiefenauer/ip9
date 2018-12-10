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
    loss_png = join(source_dir, 'loss.png')
    wer_png = join(source_dir, 'wer.png')
    ler_png = join(source_dir, 'ler.png')

    df_losses = collect_loss(source_dir)
    fig_loss, _ = plot_loss(df_losses)
    fig_loss.savefig(loss_png)
    print(f'loss plot saved to {loss_png}')

    df_metrics = collect_metrics(source_dir)
    fig_wer, *_ = plot_metric(df_metrics, 'wer')
    fig_wer.savefig(wer_png)
    print(f'WER plot saved to {wer_png}')

    fig_ler, *_ = plot_metric(df_metrics, 'ler')
    fig_ler.savefig(ler_png)
    print(f'LER plot saved to {ler_png}')

    if not silent:
        plt.show()


def collect_loss(source_dir):
    data = {}
    for tensorboard_file in sorted(glob(join(source_dir, '**/tensorboard/event*')), reverse=True):
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
        ['1_min', '10_min', '100_min', '1000_min'],
        lm_uses
    ])
    df_metric = pd.DataFrame(columns=columns)

    for subdir_name in sorted([name for name in listdir(source_dir) if isdir(join(source_dir, name))], reverse=True):
        minutes = re.findall('(\d*)', subdir_name)[0]
        metrics_csv = join(source_dir, subdir_name, f'model_{minutes}_min.csv')

        df = pd.read_csv(metrics_csv, header=[0, 1, 2], index_col=0)
        for m, d, l in itertools.product(metrics, decoding_strategies, lm_uses):
            df_metric[m, d, f'{minutes}_min', l] = df[m, d, l]
    df_metric.index += 1
    return df_metric


def plot_loss(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax_1, ax_10, ax_100, ax_1000 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    leg = ['training', 'validation']
    data_1 = df[[('CTC_train', '1_min'), ('CTC_val', '1_min')]]
    data_10 = df[[('CTC_train', '10_min'), ('CTC_val', '10_min')]]
    data_100 = df[[('CTC_train', '100_min'), ('CTC_val', '100_min')]]
    data_1000 = df[[('CTC_train', '1000_min'), ('CTC_val', '1000_min')]]
    plot_df(data_1, ax=ax_1, styles=['y-', 'y--'], legend=leg, ylabel='loss', title='1 minute')
    plot_df(data_10, ax=ax_10, styles=['g-', 'g--'], legend=leg, ylabel='loss', title='10 minutes')
    plot_df(data_100, ax=ax_100, styles=['b-', 'b--'], legend=leg, ylabel='loss', title='100 minutes')
    plot_df(data_1000, ax=ax_1000, styles=['r-', 'r--'], legend=leg, ylabel='loss', title='1000 minutes')

    fig.suptitle('CTC loss')
    # fig.tight_layout(rect=[0, 0.03, 1, 0.85])
    return fig, axes


def plot_metric(df_metrics, metric):
    metric = metric.upper()
    df_greedy = df_metrics[metric, 'greedy']
    df_beam = df_metrics[metric, 'beam']

    fig, (ax_greedy, ax_beam) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    legend = [f'{mins}{lm_use}' for mins in ['1 min', '10 mins', '100 mins', '1000 mins'] for lm_use in ['', '+LM']]
    styles = [color + line for (color, line) in itertools.product(list('ygbr'), ['-', '--'])]

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


def visualize_pipeline_performance(csv_keras, csv_ds, silent=False):
    """
    Visualize the performance of a pipeline by creating scatter plots for various correlations from a CSV file:
    - transcript length vs. LER (simplified model + DeepSpeech)
    - transcript length vs. Levenshtein similarity (simplified model + DeepSpeech)
    - transcript length vs. similarity between aligned texts of simplified and DeepSpeech model

    The plots will be saved as PNG in the same directory as the CSV

    :param csv_keras: path to CSV file containing the data
    :param csv_ds:
    :param silent: whether to show plots at the end
    """
    suptitle = 'Pipeline evaluation: Simplified Keras model vs. pre-trained DeepSpeech model'
    p_r_f_boxplot_png = join(dirname(csv_keras), f'p_r_f_boxplot.png')
    ler_similarity_png = join(dirname(csv_keras), f'similarity_scatterplot.png')

    df = pd.read_csv(csv_keras)
    keras_path = df['model path'].unique()[0]
    df['model path'] = 'Keras'

    if csv_ds:
        df_ds = pd.read_csv(csv_ds)
        ds_path = df_ds['model path'].unique()[0]
        df_ds['model path'] = 'DeepSpeech'
        df = df.append(df_ds)
    else:
        df_ds = None
        ds_path = ''

    fig, (ax_p, ax_r, ax_f) = plt.subplots(1, 3, figsize=(10, 5))

    meanprops = dict(marker='.', markeredgecolor='black', markerfacecolor='firebrick')
    df.boxplot('precision', by='model path', ax=ax_p, showmeans=True, meanprops=meanprops)
    df.boxplot('recall', by='model path', ax=ax_r, showmeans=True, meanprops=meanprops)
    df.boxplot('f-score', by='model path', ax=ax_f, showmeans=True, meanprops=meanprops)

    ax_p.set_title('Precision')
    ax_r.set_title('Recall')
    ax_f.set_title('F-Score')
    ax_p.set_xlabel('Model')
    ax_r.set_xlabel('Model')
    ax_f.set_xlabel('Model')

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.85])
    if ds_path:
        fig.text(.5, .9, f'DS model: {ds_path}', fontsize=10, ha='center')
    fig.text(.5, .85, f'Keras model: {keras_path}', fontsize=10, ha='center')
    fig.savefig(p_r_f_boxplot_png)
    print(f'saved Precision/Recall/F-Score (Boxplot) to {p_r_f_boxplot_png}')

    fig, (ax_ler, ax_sim) = plt.subplots(1, 2, figsize=(10, 5))
    df.plot.scatter(x='# words', y='LER', c='b', label='Keras', ax=ax_ler)
    if df_ds is not None:
        df_ds.plot.scatter(x='# words', y='LER', c='g', label='DeepSpeech', ax=ax_ler)

    if 'similarity' in df.keys():
        df.plot.scatter(x='# words', y='similarity', c='k', label='Keras', ax=ax_sim)

    ax_ler.set_xlabel('transcript length (characters)')
    ax_ler.set_ylabel('LER')
    ax_sim.set_xlabel('transcript length (characters)')
    ax_sim.set_ylabel('Levensthein similarity')

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.85])
    fig.text(.5, .9, f'DS model: {ds_path}', fontsize=10, ha='center')
    fig.text(.5, .85, f'Keras model: {keras_path}', fontsize=10, ha='center')
    fig.savefig(ler_similarity_png)
    print(f'saved LER similarity (Scatterplot) to {ler_similarity_png}')

    if not silent:
        plt.show()

    return p_r_f_boxplot_png, ler_similarity_png
