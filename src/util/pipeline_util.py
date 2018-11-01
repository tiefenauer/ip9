"""
Utility functions for end-to-end tasks
"""
import json
import os
from os.path import join, basename, exists, pardir, abspath
from shutil import copyfile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pattern3.metrics import levenshtein_similarity

from constants import ASSETS_DIR
from util.audio_util import frame_to_ms
from util.lm_util import ler_norm
from util.rnn_util import query_gpu


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
    create_demo_index(target_dir, demo_id, audio_src_path, transcript, df_transcripts, df_stats)

    assets_dir = join(ASSETS_DIR, 'demo')
    for file in [file for _, _, files in os.walk(assets_dir) for file in files]:
        copyfile(join(assets_dir, file), join(target_dir, file))
    copyfile(join(ASSETS_DIR, 'start_server.sh'), join(join(target_dir, pardir), 'start_server.sh'))


def create_alignment_json(df_transcripts):
    alignments = [{'id': ix,
                   'transcript': row['transcript'],
                   'text': row['alignment'],
                   'start': row['audio_start'],
                   'end': row['audio_end']
                   } for ix, row in df_transcripts.iterrows()]
    return {'alignments': alignments}


def create_demo_index(target_dir, demo_id, audio_src_path, transcript, df_transcripts, df_stats):
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
    n_aligned = len(' '.join([t for t in df_transcripts['alignment'] if t]))
    metrics_table = soup.find(id='metrics')
    metrics_table.append(create_tr('directory', target_dir))
    metrics_table.append(create_tr('audio file', audio_src_path))
    metrics_table.append(create_tr('transcript length', f'{n_chars} characters, {len(transcript.split())} words'))
    metrics_table.append(create_tr('#aligned chars', f'{n_aligned} ({100*n_aligned/n_chars:.2f}%)'))
    metrics_table.append(create_tr('#alignments/segments', f'{len(df_transcripts)}'))

    for ix, (ler, similarity) in df_stats.iterrows():
        metrics_table.append(create_tr('Ø LER', ler))
        metrics_table.append(create_tr('Ø similarity inference/alignment', similarity))

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
    if not args.keras_path and not args.ds_path:
        args.keras_path = input('Enter path to directory containing Keras model (*.h5) or leave blank to use DS: ')
    if args.keras_path:
        keras_path = abspath(args.keras_path)
        if not exists(keras_path):
            raise ValueError(f'ERROR: Keras model not found at {keras_path}')

    ds_path, ds_alpha_path, ds_trie_path = None, None, None
    if not keras_path and not args.ds_path:
        while not args.ds_path:
            args.ds_path = input('Enter path to DeepSpeech model (*.pbmm): ')
        while not args.ds_alpha_path:
            args.ds_alpha_path = input('Enter path to alphabet file (*.txt): ')
        while not args.ds_trie_path:
            args.ds_trie_path = input('Enter path to trie file: ')
    if args.ds_path:
        ds_path = abspath(args.ds_path)
        if not exists(ds_path):
            raise ValueError(f'ERROR: DS model not found at {ds_path}')

        if not args.ds_alpha_path:
            raise ValueError('ERROR: alphabet path must be specified when using DeepSpeech model')
        ds_alpha_path = abspath(args.ds_alpha_path)
        if not exists(ds_alpha_path):
            raise ValueError(f'ERROR: alphabet not found at {ds_alpha_path}')

        if not args.ds_trie_path:
            raise ValueError('ERROR: trie must be specified when using DeepSpeech model')
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
