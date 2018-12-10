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
    with open(alignment_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f'saved alignment information to {alignment_json_path}')

    demo_id = basename(target_dir)
    add_demo_to_index(target_dir, demo_id, df_stats)
    create_demo_index(target_dir, demo_id, audio_src_path, transcript, df_stats)

    assets_dir = join(ASSETS_DIR, 'demo')
    for file in [file for _, _, files in os.walk(assets_dir) for file in files]:
        copyfile(join(assets_dir, file), join(target_dir, file))
    copyfile(join(ASSETS_DIR, 'start_server.sh'), join(join(target_dir, pardir), 'start_server.sh'))


def create_alignment_json(df_transcripts):
    alignments = [{'id': ix,
                   'transcript': row['transcript'],
                   'text': row['alignment'],
                   'audio_start': row['audio_start'],
                   'audio_end': row['audio_end'],
                   'text_start': row['text_start'],
                   'text_end': row['text_end']
                   } for ix, row in df_transcripts.iterrows()]
    return {'alignments': alignments}


def create_demo_index(target_dir, demo_id, audio_src_path, transcript, df_stats):
    template_path = join(ASSETS_DIR, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = demo_id
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript

    def create_tr(*args):
        tr = soup.new_tag('tr')
        for arg in args:
            td = soup.new_tag('td')
            td.string = str(arg)
            tr.append(td)
        return tr

    metrics_table = soup.find(id='metrics')
    metrics_table.append(create_tr('directory', target_dir))
    metrics_table.append(create_tr('audio file', audio_src_path))

    for column in df_stats:
        metrics_table.append(create_tr(column, df_stats.loc[0, column]))

    demo_index_path = join(target_dir, 'index.html')
    with open(demo_index_path, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_index_path


def add_demo_to_index(target_dir, demo_id, df_stats):
    index_path = join(join(target_dir, pardir), 'index.html')
    if not exists(index_path):
        copyfile(join(ASSETS_DIR, '_index_template.html'), index_path)

    soup = BeautifulSoup(open(index_path), 'html.parser')
    table = soup.find(id='demo_list')

    if not soup.find(id=demo_id):
        tr = soup.new_tag('tr', id=demo_id)

        a = soup.new_tag('a', href=demo_id)
        a.string = demo_id
        td = soup.new_tag('td')
        td.append(a)
        tr.append(td)

        precision = df_stats.loc[0, 'precision']
        td = soup.new_tag('td')
        td.string = f'{precision:.4f}'
        tr.append(td)

        recall = df_stats.loc[0, 'recall']
        td = soup.new_tag('td')
        td.string = f'{recall:.4f}'
        tr.append(td)

        f_score = df_stats.loc[0, 'f-score']
        td = soup.new_tag('td')
        td.string = f'{f_score:.4f}'
        tr.append(td)

        ler = df_stats.loc[0, 'LER']
        td = soup.new_tag('td')
        td.string = f'{ler:.4f}'
        tr.append(td)

        similarity = df_stats.loc[0, 'similarity']
        td = soup.new_tag('td')
        td.string = f'{similarity:.4f}'
        tr.append(td)

        table.append(tr)

        with open(index_path, 'w') as f:
            f.write(soup.prettify())


def update_index(target_dir, lang, num_aligned, df_keras=None, keras_path=None, df_ds=None, ds_path=None, lm_path=None,
                 vocab_path=None):
    index_path = join(target_dir, 'index.html')
    soup = BeautifulSoup(open(index_path), 'html.parser')
    soup.find(id='title').string = f'Forced Alignment Demo ({lang})'

    soup.find(id='num_aligned').string = str(num_aligned)
    soup.find(id='keras_path').string = keras_path if keras_path else ''
    soup.find(id='ds_path').string = ds_path if ds_path else ''
    soup.find(id='lm_path').string = lm_path if lm_path else ''
    soup.find(id='vocab_path').string = vocab_path if vocab_path else ''

    if df_keras is not None:
        av_p = df_keras['precision'].mean()
        av_r = df_keras['recall'].mean()
        av_f = df_keras['f-score'].mean()
        soup.find(id='precision_keras').string = f'{av_p:.4f}'
        soup.find(id='recall_keras').string = f'{av_r:.4f}'
        soup.find(id='f-score_keras').string = f'{av_f:.4f}'

    if df_ds is not None:
        av_p = df_ds['precision'].mean()
        av_r = df_ds['recall'].mean()
        av_f = df_ds['f-score'].mean()
        soup.find(id='precision_ds').string = f'{av_p:.4f}'
        soup.find(id='recall_ds').string = f'{av_r:.4f}'
        soup.find(id='f-score_ds').string = f'{av_f:.4f}'

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

    return keras_path, ds_path, ds_alpha_path, ds_trie_path


def query_lm_params(args):
    if not args.lm_path:
        args.lm_path = input('Enter path to LM to use for spell checking (enter nothing for no spell checking): ')
        if args.lm_path:
            if not exists(abspath(args.lm_path)):
                raise ValueError(f'ERROR: LM not found at {abspath(args.lm_path)}!')
            if not args.vocab_path:
                args.vocab_path = input('Enter path to vocabulary file to use for spell checker: ')
                if args.vocab_path:
                    if not exists(abspath(args.vocab_path)):
                        raise ValueError(f'ERROR: Vocabulary not found at {abspath(args.vocab_path)}!')

    lm_path = abspath(args.lm_path) if args.lm_path else ''
    vocab_path = abspath(args.vocab_path) if args.vocab_path else ''
    return lm_path, vocab_path


def calculate_stats(df_alignments, model_path, transcript):
    partial_transcripts = df_alignments['transcript'].values
    alignments = df_alignments['alignment'].values

    # Precision = similarity between transcript and alignment
    p = np.mean([levenshtein_similarity(t, a) for t, a in zip(partial_transcripts, alignments)])
    # Recall = fraction of aligned text
    merged_alignments = ' '.join(a for a in alignments if a)
    r = len(merged_alignments) / len(transcript)
    # F-Score
    f = 2 * p * r / (p + r)

    ler_avg = np.mean([ler_norm(gt, al) for gt, al in zip(partial_transcripts, alignments)])

    data = [[model_path, len(alignments), len(transcript.split()), len(transcript), p, r, f, ler_avg]]
    columns = ['model path', '# alignments', '# words', '# characters', 'precision', 'recall', 'f-score', 'LER']
    return pd.DataFrame(data, columns=columns)
