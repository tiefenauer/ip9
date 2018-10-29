# Create ReadyLingua Corpus
import argparse
import logging
import math
import os
import sys
import wave
from os import makedirs, walk
from os.path import exists, join
from pathlib import Path

from lxml import etree
from pydub.utils import mediainfo
from tqdm import tqdm

from constants import RL_TARGET, RL_SOURCE
from corpus.corpus import ReadyLinguaCorpus
from corpus.corpus_entry import CorpusEntry
from corpus.corpus_segment import SpeechSegment
from util.audio_util import resample_frame, to_wav
from util.corpus_util import find_file_by_suffix
from util.log_util import log_setup, create_args_str
from util.string_util import create_filename

logfile = 'create_rl_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
LANGUAGES = {  # mapping from folder names to language code
    'Deutsch': 'de',
    'Englisch': 'en',
    'Französisch': 'fr',
    'Italienisch': 'it',
    'Spanisch': 'es'
}

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Other values
# -------------------------------------------------------------
# currently none

def main():
    print(create_args_str(args))
    print(f'Processing files from {args.source_root} and saving them in {args.target_root}')
    corpus, corpus_file = create_corpus(args.source_root, args.target_root, args.max_entries)
    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')


def create_corpus(source_root, target_root, max_entries=None):
    if not exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not exists(target_root):
        makedirs(target_root)

    return create_readylingua_corpus(source_root, target_root, max_entries)


def create_readylingua_corpus(source_root, target_root, max_entries):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    log.info('Collecting files')
    corpus_entries = []

    directories = [root for root, subdirs, files in walk(source_root)
                   if not subdirs  # only include leaf directories
                   and not root.endswith(os.sep + 'old')  # '/old' leaf-folders are considered not reliable
                   and not os.sep + 'old' + os.sep in root]  # also exclude /old/ non-leaf folders

    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')

    for raw_path in progress:
        if max_entries and len(corpus_entries) >= max_entries:
            break

        progress.set_description(f'{raw_path:{100}}')

        files = collect_files(raw_path)
        if not files:
            log.warning(f'Skipping directory (not all files found): {raw_path}')
            continue

        parms = collect_corpus_entry_parms(raw_path, files)

        segmentation_file = join(raw_path, files['segmentation'])
        index_file = join(raw_path, files['index'])
        transcript_file = join(raw_path, files['text'])

        segments = create_segments(index_file, transcript_file, parms['rate'], parms['language'])

        # Resample and crop audio
        audio_dst = join(target_root, parms['id'] + ".wav")
        if not exists(audio_dst) or args.overwrite:
            audio_src = join(raw_path, files['audio'])
            to_wav(audio_src, audio_dst)
            # audio, rate = librosa.load(audio_src, sr=16000, mono=True)
            # audio, rate, segments, crop_to_segments(audio, rate, segments)
            # sf.write(audio_dst, audio, rate, subtype='PCM_16')
        parms['media_info'] = mediainfo(audio_dst)

        # Create corpus entry
        corpus_entry = CorpusEntry(audio_dst, segments, raw_path=raw_path, parms=parms)
        corpus_entries.append(corpus_entry)

    corpus = ReadyLinguaCorpus(corpus_entries)
    corpus_file = save_corpus(corpus, target_root)
    return corpus, corpus_file


def collect_files(directory):
    project_file = find_file_by_suffix(directory, ' - Project.xml')
    if project_file:
        audio_file, text_file, segmentation_file, index_file = parse_project_file(join(directory, project_file))
    else:
        audio_file, text_file, segmentation_file, index_file = scan_content_dir(directory)

    files = {'audio': audio_file, 'text': text_file, 'segmentation': segmentation_file, 'index': index_file}

    # check if all file names were found
    for attribute_name, file_name in files.items():
        if not file_name:
            log.warning(f'File \'{attribute_name}\' is not set')
            return None

    # check if files exist
    for file_name in files.values():
        path = Path(directory, file_name)
        if not path.exists():
            log.warning(f'File does not exist: {path}')
            return None

    return files


def parse_project_file(project_file):
    doc = etree.parse(project_file)
    for element in ['AudioFiles/Name', 'TextFiles/Name', 'SegmentationFiles/Name', 'IndexFiles/Name']:
        if doc.find(element) is None:
            log.warning(f'Invalid project file (missing element \'{element}\'): {project_file}')
            return None, None, None, None

    audio_file = doc.find('AudioFiles/Name').text
    text_file = doc.find('TextFiles/Name').text
    segmentation_file = doc.find('SegmentationFiles/Name').text
    index_file = doc.find('IndexFiles/Name').text
    return audio_file, text_file, segmentation_file, index_file


def scan_content_dir(content_dir):
    audio_file = find_file_by_suffix(content_dir, '.wav')
    text_file = find_file_by_suffix(content_dir, '.txt')
    segmentation_file = find_file_by_suffix(content_dir, ' - Segmentation.xml')
    index_file = find_file_by_suffix(content_dir, ' - Index.xml')
    return audio_file, text_file, segmentation_file, index_file


def collect_corpus_entry_parms(directory, files):
    index_file_path = join(directory, files['index'])
    audio_file_path = join(directory, files['audio'])
    folders = directory.split(os.sep)

    # find name and create id
    corpus_entry_name = folders[-1]
    corpus_entry_id = create_filename(files['audio'].split('.wav')[0])

    # find language
    lang = [folder for folder in folders if folder in LANGUAGES.keys()]
    language = LANGUAGES[lang[0]] if lang else 'unknown'

    # find sampling rate
    doc = etree.parse(index_file_path)
    rate = int(doc.find('SamplingRate').text)

    # find number of channels
    channels = wave.open(audio_file_path, 'rb').getnchannels()

    return {'id': corpus_entry_id, 'name': corpus_entry_name, 'language': language, 'rate': rate, 'channels': channels}


def create_segments(index_file, transcript_file, original_sample_rate, language):
    # segmentation = collect_segmentation(segmentation_file)
    speeches = collect_speeches(index_file)
    transcript = Path(transcript_file).read_text(encoding='utf-8')

    # merge information from index file (speech parts) with segmentation information
    segments = []
    for speech_meta in speeches:
        # re-calculate original frame values to resampled values
        start_frame = resample_frame(speech_meta['start_frame'], old_sampling_rate=original_sample_rate)
        end_frame = resample_frame(speech_meta['end_frame'], old_sampling_rate=original_sample_rate)
        start_text = speech_meta['start_text']
        end_text = speech_meta['end_text'] + 1  # komische Indizierung

        seg_transcript = transcript[start_text:end_text]
        segments.append(SpeechSegment(start_frame=start_frame,
                                      end_frame=end_frame,
                                      transcript=seg_transcript,
                                      language=language))

    return segments


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

        speech = {'start_frame': start_frame, 'end_frame': end_frame, 'start_text': start_text, 'end_text': end_text}
        speeches.append(speech)
    return sorted(speeches, key=lambda s: s['start_frame'])


if __name__ == '__main__':
    main()
