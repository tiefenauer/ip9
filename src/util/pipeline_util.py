"""
Utility functions for end-to-end tasks
"""
import json
import os
from os.path import join, relpath, basename, exists, pardir, dirname
from pathlib import Path
from shutil import copyfile

from bs4 import BeautifulSoup

from constants import ROOT_DIR

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


def create_demo(target_dir, audio_src_path, transcript, df_transcripts, df_stats, demo_id=None):
    if not demo_id:
        demo_id = basename(audio_src_path)
    print(f'assigned demo id: {demo_id}.')

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
    soup.title.string = 'Some Title'
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript.replace('\n', ' ')

    def create_tr(*args):
        tr = soup.new_tag('tr')
        for arg in args:
            td = soup.new_tag('td')
            td.string = arg
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

    for ix, (metric, value) in df_stats.iterrows():
        metrics_table.append(create_tr(metric, value))

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
