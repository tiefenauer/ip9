# Create ReadyLingua Corpus
import argparse
import math
import os
import sys
from collections import Counter
from os import makedirs, walk
from os.path import exists, join, splitext, basename
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

from constants import RL_TARGET, RL_SOURCE
from util.audio_util import resample_frame, crop_and_resample
from util.corpus_util import find_file_by_suffix
from util.log_util import create_args_str
from util.string_util import create_filename, normalize, contains_numeric

LANGUAGES = {  # mapping from folder names to language code
    'Deutsch': 'de',
    'Englisch': 'en',
    'Französisch': 'fr',
    'Italienisch': 'it',
    'Spanisch': 'es'
}

parser = argparse.ArgumentParser(description="""Create ReadyLingua corpus from raw files""")
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-s', '--source_root', default=RL_SOURCE,
                    help=f'(optional) source root directory (default: {RL_SOURCE}')
parser.add_argument('-t', '--target_root', default=RL_TARGET,
                    help=f'(optional) target root directory (default: {RL_TARGET})')
parser.add_argument('-m', '--max_entries', type=int, default=None,
                    help='(optional) maximum number of corpus entries to process. Default=None=\'all\'')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing audio data if already present. Default=False)')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    print(f'Processing files from {args.source_root} and saving them in {args.target_root}')
    corpus, corpus_file = create_corpus(args.source_root, args.target_root, args.max_entries)
    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')


def create_corpus(source_dir, target_dir, max_entries=None):
    if not exists(source_dir):
        print(f"ERROR: Source root {source_dir} does not exist!")
        exit(0)
    if not exists(target_dir):
        makedirs(target_dir)

    df = create_segments(source_dir, target_dir, max_entries)
    index_file = join(target_dir, 'index.csv')
    df.to_csv(index_file)

    return df, index_file


def create_segments(source_dir, target_dir, max_entries):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    print('Collecting files')

    directories = [root for root, subdirs, files in walk(source_dir)
                   if not subdirs  # only include leaf directories
                   and not root.endswith(os.sep + 'old')  # '/old' leaf-folders are considered not reliable
                   and not os.sep + 'old' + os.sep in root][:max_entries]  # also exclude /old/ non-leaf folders

    segments = []
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')
    for source_dir in progress:
        progress.set_description(f'{source_dir:{100}}')

        audio_file, transcript_file, segmentation_file, index_file = collect_files(source_dir)

        if not all(file is not None and len(file.strip()) > 0 for file in
                   [audio_file, transcript_file, segmentation_file, index_file]):
            print(f'Skipping directory (not all files found): {source_dir}')
            continue

        entry_id, entry_name, lang, rate = collect_corpus_entry_parms(source_dir, index_file, audio_file)

        segment_infos = extract_segment_infos(index_file, transcript_file, rate, lang)

        wav_file = join(target_dir, entry_id + ".wav")
        if not exists(wav_file) or args.overwrite:
            crop_and_resample(audio_file, wav_file, segment_infos)

        # write full transcript
        with open(transcript_file) as f_src, open(join(target_dir, f'{entry_id}.txt'), 'w') as f_dst:
            transcript = normalize(f_src.read(), lang)
            f_dst.write(transcript)

        # create segment
        for segment_info in segment_infos:
            subset = 'n/a'  # must be set after all segments have been processed
            audio_file = basename(wav_file)
            start_frame = segment_info['start_frame']
            end_frame = segment_info['end_frame']
            transcript = segment_info['transcript']
            duration = (end_frame - start_frame) / 16000
            numeric = contains_numeric(transcript)
            segments.append([entry_id, subset, lang, audio_file, start_frame, end_frame, duration, transcript, numeric])

    columns = ['entry_id', 'subset', 'language', 'audio_file', 'start_frame', 'end_frame', 'duration', 'transcript', 'numeric']
    df = pd.DataFrame(segments, columns=columns)

    """
    because ReadyLingua data is not pre-partitioned into train-/dev-/test-data this needs to be done after all
    corpus entries and segments are known
    """
    total_audio = df.groupby('language')['duration'].sum().to_dict()
    audio_per_language = Counter()
    for (id, lang), df_entry in df.groupby(['entry_id', 'language']):
        if audio_per_language[lang] > 0.9 * total_audio[lang]:
            subset = 'test'
        elif audio_per_language[lang] > 0.8 * total_audio[lang]:
            subset = 'dev'
        else:
            subset = 'train'
        df.loc[df['entry_id'] == id, 'subset'] = subset
        audio_per_language[lang] += df_entry['duration'].sum()

    return df


def collect_files(source_dir):
    project_file = find_file_by_suffix(source_dir, ' - Project.xml')
    if project_file:
        audio_file, transcript_file, segmentation_file, index_file = parse_project_file(join(source_dir, project_file))
    else:
        audio_file, transcript_file, segmentation_file, index_file = scan_content_dir(source_dir)

    # check if files are set
    if not audio_file:
        print('WARNING: audio file is not set')
        return None, None, None, None
    if not transcript_file:
        print('WARNING: transcript file is not set')
        return None, None, None, None
    if not segmentation_file:
        print('WARNING: segmentation file is not set')
        return None, None, None, None
    if not index_file:
        print('WARNING: index file is not set')
        return None, None, None, None

    audio_file = join(source_dir, audio_file)
    transcript_file = join(source_dir, transcript_file)
    segmentation_file = join(source_dir, segmentation_file)
    index_file = join(source_dir, index_file)

    # check if files exist
    if not exists(audio_file):
        print(f'WARNING: file {audio_file} does not exist')
        return None, None, None, None
    if not exists(transcript_file):
        print(f'WARNING: file {transcript_file} does not exist')
        return None, None, None, None
    if not exists(segmentation_file):
        print(f'WARNING: file {segmentation_file} does not exist')
        return None, None, None, None
    if not exists(index_file):
        print(f'WARNING: file {index_file} does not exist')
        return None, None, None, None

    return audio_file, transcript_file, segmentation_file, index_file


def parse_project_file(project_file):
    doc = etree.parse(project_file)
    for element in ['AudioFiles/Name', 'TextFiles/Name', 'SegmentationFiles/Name', 'IndexFiles/Name']:
        if doc.find(element) is None:
            print(f'Invalid project file (missing element \'{element}\'): {project_file}')
            return None, None, None, None

    audio_file = doc.find('AudioFiles/Name').text
    transcript_file = doc.find('TextFiles/Name').text
    segmentation_file = doc.find('SegmentationFiles/Name').text
    index_file = doc.find('IndexFiles/Name').text
    return audio_file, transcript_file, segmentation_file, index_file


def scan_content_dir(content_dir):
    audio_file = find_file_by_suffix(content_dir, '.wav')
    text_file = find_file_by_suffix(content_dir, '.txt')
    segmentation_file = find_file_by_suffix(content_dir, ' - Segmentation.xml')
    index_file = find_file_by_suffix(content_dir, ' - Index.xml')
    return audio_file, text_file, segmentation_file, index_file


def collect_corpus_entry_parms(directory, index_file, audio_file):
    entry_name = basename(directory)
    entry_id = create_filename(splitext(basename(audio_file))[0])

    # find language
    lang = [folder for folder in directory.split(os.sep) if folder in LANGUAGES.keys()]
    language = LANGUAGES[lang[0]] if lang else 'unknown'

    # find sampling rate
    doc = etree.parse(index_file)
    rate = int(doc.find('SamplingRate').text)

    return entry_id, entry_name, language, rate


def extract_segment_infos(index_file, transcript_file, src_rate, language):
    # segmentation = collect_segmentation(segmentation_file)
    speeches = collect_speeches(index_file)
    transcript = Path(transcript_file).read_text(encoding='utf-8')

    # merge information from index file (speech parts) with segmentation information
    segment_infos = []
    for speech_meta in speeches:
        start_text = speech_meta['start_text']
        end_text = speech_meta['end_text'] + 1  # komische Indizierung

        segment_infos.append({
            'start_frame': resample_frame(speech_meta['start_frame'], src_rate=src_rate),
            'end_frame': resample_frame(speech_meta['end_frame'], src_rate=src_rate),
            'transcript': normalize(transcript[start_text:end_text], language)
        })

    return segment_infos


def collect_segmentation(segmentation_file):
    segments = []
    doc = etree.parse(segmentation_file)
    for element in doc.findall('Segments/Segment'):
        start_frame = int(element.attrib['start'])
        end_frame = int(element.attrib['end'])
        segment = {'class': element.attrib['class'], 'start_frame': start_frame, 'end_frame': end_frame}
        segments.append(segment)

    return sorted(segments, key=lambda s: s['start_frame'])


def collect_speeches(index_file):
    speeches = []
    doc = etree.parse(index_file)
    for element in doc.findall('TextAudioIndex'):
        start_text = int(element.find('TextStartPos').text)
        end_text = int(element.find('TextEndPos').text)
        start_frame = int(element.find('AudioStartPos').text)
        end_frame = int(element.find('AudioEndPos').text)

        speech = {'start_frame': start_frame, 'end_frame': end_frame, 'start_text': start_text,
                  'end_text': end_text}
        speeches.append(speech)
    return sorted(speeches, key=lambda s: s['start_frame'])


def get_index_for_audio_length(segments, min_length):
    """
    get index to split speech segments at a minimum audio length.Index will not split segments of same corpus entry
    :param segments: list of speech segments
    :param min_length: minimum audio length to split
    :return: first index where total length of speech segments is equal or greater to minimum legnth
    """
    audio_length = 0
    prev_corpus_entry_id = None
    for ix, segment in enumerate(segments):
        audio_length += segment.audio_length
        if audio_length > min_length and segment.corpus_entry.id is not prev_corpus_entry_id:
            return ix
        prev_corpus_entry_id = segment.corpus_entry.id


if __name__ == '__main__':
    main()
