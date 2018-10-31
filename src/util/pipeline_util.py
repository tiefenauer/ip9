"""
Utility functions for end-to-end tasks
"""
import json
import os
from os.path import join, relpath, basename, exists, pardir, dirname, abspath
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pattern3.metrics import levenshtein_similarity

from constants import ROOT_DIR
from util.audio_util import frame_to_ms
from util.lm_util import ler_norm
from util.rnn_util import query_gpu

ASSETS_DIR = join(ROOT_DIR, 'assets')
ASSETS_FILES = [
    'aligner.js',
    'default.css',
    'style.css',
    'jquery-3.3.2.min.js',
    'bootstrap.bundle.min.js',
    'bootstrap.bundle.min.js.map',
    # 'bootstrap-popover.css',
    'bootstrap-tooltip.css',
    # 'bootstrap.min.css',
    # 'bootstrap.min.css.map'
]


def create_demo_files(target_dir, audio_src_path, transcript, df_transcripts, df_stats):
    audio_dst_path = join(target_dir, 'audio.mp3')
    copyfile(audio_src_path, audio_dst_path)
    print(f'saved audio to {audio_dst_path}')

    transcript_path = join(target_dir, 'transcript.txt')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f'saved transcript to {transcript_path}')

    json_data = create_alignment_json(df_transcripts)
    alignment_json_path = join(target_dir, 'alignment.json')
    with open(alignment_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'saved alignment information to {alignment_json_path}')

    demo_id = basename(target_dir)
    update_index(target_dir, demo_id)
    demo_path = create_demo_index(target_dir, demo_id, audio_dst_path, transcript_path, transcript, df_transcripts,
                                  df_stats)
    for file in ASSETS_FILES:
        copyfile(join(ASSETS_DIR, file), join(target_dir, file))
    copyfile(join(ASSETS_DIR, 'start_server.sh'), join(join(target_dir, pardir), 'start_server.sh'))
    return create_url(demo_path, target_dir)


def create_alignment_json(df_transcripts):
    alignments = [{'id': ix,
                   'transcript': row['transcript'],
                   'text': row['alignment'],
                   'start': row['audio_start'],
                   'end': row['audio_end']
                   } for ix, row in df_transcripts.iterrows()]
    return {'alignments': alignments}


def create_demo_index(target_dir, demo_id, audio_path, transcript_path, transcript, df_transcripts, df_stats):
    template_path = join(ASSETS_DIR, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = demo_id
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript.replace('\n', ' ')

    def create_tr(*args):
        tr = soup.new_tag('tr')
        for arg in args:
            td = soup.new_tag('td')
            td.string = str(arg)
            tr.append(td)
        return tr

    n_chars = len(transcript)
    n_words = len(transcript.split())
    n_alignments = len(df_transcripts)
    n_aligned = len(' '.join(df_transcripts['alignment']))
    metrics_table = soup.find(id='metrics')
    metrics_table.append(create_tr('directory', dirname(audio_path)))
    metrics_table.append(create_tr('audio', basename(audio_path)))
    metrics_table.append(create_tr('transcript', basename(transcript_path)))
    metrics_table.append(create_tr('transcript length', f'{n_chars} characters, {n_words} words'))
    metrics_table.append(create_tr('#aligned chars', f'{n_aligned} ({100*n_aligned/n_chars:.3f}%)'))
    metrics_table.append(create_tr('#alignments', f'{n_alignments}'))

    for ix, (ler, similarity) in df_stats.iterrows():
        metrics_table.append(create_tr('Ø LER', ler))
        metrics_table.append(create_tr('Ø similarity', similarity))

    demo_index_path = join(target_dir, 'index.html')
    with open(demo_index_path, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_index_path


def update_index(target_dir, demo_id):
    index_path = join(join(target_dir, pardir), 'index.html')
    if not exists(index_path):
        copyfile(join(ASSETS_DIR, '_index_template.html'), index_path)

    soup = BeautifulSoup(open(index_path), 'html.parser')

    if not soup.find(id=demo_id):
        a = soup.new_tag('a', href=demo_id)
        a.string = demo_id
        li = soup.new_tag('li', id=demo_id)
        li.append(a)
        ul = soup.find(id='demo_list')
        ul.append(li)

        with open(index_path, 'w') as f:
            f.write(soup.prettify())


def create_url(demo_path, target_dir):
    return 'https://ip8.tiefenauer.info:8888/' + relpath(demo_path, Path(target_dir).parent).replace(os.sep, '/')


def create_alignments_dataframe(voiced_segments, transcripts, sample_rate):
    alignments = []
    for i, (voice_segment, transcript) in enumerate(zip(voiced_segments, transcripts)):
        audio_start = frame_to_ms(voice_segment.start_frame, sample_rate)
        audio_end = frame_to_ms(voice_segment.end_frame, sample_rate)
        alignments.append([transcript, audio_start, audio_end])

    df_alignments = pd.DataFrame(alignments, columns=['transcript', 'audio_start', 'audio_end'])
    df_alignments.index.name = 'id'
    return df_alignments


def query_asr_params(args):
    """
    Helper function to query ASR model from user if not set in args
    """
    gpu = query_gpu(args.gpu)

    keras_path = None
    if not not args.keras_path and not args.ds_path:
        args.keras_path = input('Enter path to directory containing Keras model (*.h5) or leave blank to use DS: ')
    if args.keras_path:
        keras_path = abspath(args.keras_path)
        if not exists(keras_path):
            raise ValueError(f'ERROR: Keras model not found at {keras_path}')

    if not keras_path and not args.ds_path:
        while not args.ds_path:
            args.ds_path = input('Enter path to directory containing DeepSpeech model (*.pbmm): ')
    ds_path = abspath(args.ds_path)

    if not exists(ds_path):
        raise ValueError(f'ERROR: DeepSpeech model not found at {ds_path}')

    if not args.ds_alpha_path:
        raise ValueError('ERROR: alphabet path must be specified when using DeepSpeech model')

    ds_alpha_path = abspath(args.ds_alpha_path)
    if not exists(ds_alpha_path):
        raise ValueError(f'ERROR: alphabet not found at {ds_alpha_path}')

    ds_trie_path = abspath(args.ds_trie_path)
    if not exists(ds_trie_path):
        raise ValueError(f'ERROR: Trie not found at {ds_trie_path}')

    while not args.lm_path:
        args.lm_path = input('Enter path to binary file of KenLM n-gram model. Leave blank for no LM: ')
        if args.lm_path and not exists(abspath(args.lm_path)):
            raise ValueError(f'ERROR: LM not found at {abspath(args.lm_path)}')

    lm_path = abspath(args.lm_path)
    return keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, gpu


def calculate_stats(df_alignments):
    ground_truths = df_alignments['transcript'].values
    alignments = df_alignments['alignment'].values

    ler_avg = np.mean([ler_norm(gt, al) for gt, al in zip(ground_truths, alignments)])
    similarity_avg = np.mean([levenshtein_similarity(gt, al) for gt, al in zip(ground_truths, alignments)])

    return pd.DataFrame([[ler_avg, similarity_avg]], columns=['LER', 'similarity'])
