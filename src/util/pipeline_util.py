"""
Utility functions for end-to-end tasks
"""
import json
import os
from os.path import join, relpath, splitext, basename
from pathlib import Path
from shutil import copyfile

import langdetect
from bs4 import BeautifulSoup

from constants import ROOT_DIR

ASSETS_DIR = join(ROOT_DIR, 'assets')


def create_demo(target_dir, audio_src_path, transcript, df_transcripts, df_stats, demo_id=None):
    if not demo_id:
        demo_id = basename(audio_src_path)
    print(f'assigned demo id: {demo_id}.')

    language = langdetect.detect(transcript)
    print(f'detected language: {language}')

    audio_dst_path = join(target_dir, 'audio.mp3')
    transcript_path = join(target_dir, 'transcript.txt')
    alignment_json_path = join(target_dir, 'alignment.json')

    print(f'saving audio in {audio_dst_path}')
    copyfile(audio_src_path, audio_dst_path)

    print(f'saving transcript in {transcript_path}')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f'saving alignment information to {alignment_json_path}')
    json_data = create_alignment_json(df_transcripts)
    with open(alignment_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # update_index(demo_id)
    demo_path = create_demo_index(target_dir, demo_id, transcript)
    for file in ['aligner.js', 'default.css', 'style.css', 'server.py', 'jquery-3.3.2.min.js']:
        copyfile(join(ASSETS_DIR, file), join(target_dir, file))
    return create_url(demo_path, target_dir)


def create_alignment_json(df_transcripts):
    alignments = []
    for _, row in df_transcripts.iterrows():
        text = row['alignment']
        audio_start = row['audio_start']
        audio_end = row['audio_end']
        alignments.append([text, audio_start, audio_end])
    return {'alignments': alignments}


def create_demo_index(target_dir, demo_id, transcript):
    template_path = join(ASSETS_DIR, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = 'Some Title'
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript.replace('\n', ' ')

    demo_path = join(target_dir, 'index.html')
    with open(demo_path, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_path


def update_index(demo_id):
    index_path = join(ASSETS_DIR, 'index.html')
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
