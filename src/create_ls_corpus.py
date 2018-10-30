# Create ReadyLingua Corpus
import argparse
import math
import re
import sys
from os import makedirs, walk
from os.path import exists, splitext, join, basename, pardir, abspath

from tabulate import tabulate
from tqdm import tqdm
from unidecode import unidecode

from constants import LS_SOURCE, LS_TARGET
from corpus.corpus import LibriSpeechCorpus
from corpus.corpus_entry import CorpusEntry
from corpus.corpus_segment import Segment
from util.audio_util import seconds_to_frame, crop_and_resample
from util.corpus_util import find_file_by_suffix, save_corpus
from util.log_util import create_args_str
from util.string_util import normalize

parser = argparse.ArgumentParser(description="""Create LibriSpeech corpus from raw files""")
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-s', '--source', default=LS_SOURCE,
                    help=f'(optional) source root directory (default: {LS_SOURCE}')
parser.add_argument('-t', '--target', default=LS_TARGET,
                    help=f'(optional) target root directory (default: {LS_TARGET})')
parser.add_argument('-m', '--max_entries', type=int, default=None,
                    help='(optional) maximum number of corpus entries to process. Default=None=\'all\'')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing audio data if already present. If set to true this will '
                         'convert, resample and crop the audio data to a 16kHz mono WAV file which will prolong the'
                         'corpus creation process considerably. If set to false, the conversion of audio data will be'
                         'skipped, if the file is already present in the target directory and the corpus will only be'
                         'updated with the most current corpus entries. Default=False)')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    print(f'Processing files from {args.source} and saving them in {args.target}')
    corpus, corpus_file = create_corpus(args.source, args.target, args.max_entries)
    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')


def create_corpus(source_dir, target_dir, max_entries=None):
    if not exists(source_dir):
        print(f"ERROR: Source directory {source_dir} does not exist!")
        exit(0)
    if not exists(target_dir):
        print(f'creating target directory {target_dir} as it does not exist yet')
        makedirs(target_dir)

    entries = create_entries(source_dir=source_dir, target_dir=target_dir, max_entries=max_entries)

    corpus = LibriSpeechCorpus(entries)
    corpus_index = save_corpus(corpus, target_dir)
    return corpus, corpus_index


def create_entries(source_dir, target_dir, max_entries):
    audio_root = join(source_dir, 'audio')
    books_root = join(source_dir, 'books')

    chapters_file = find_file_by_suffix(audio_root, 'CHAPTERS.TXT')
    chapter_meta = collect_chapter_meta(chapters_file)

    book_texts = collect_book_texts(books_root)

    directories = [root for root, subdirs, files in walk(audio_root) if not subdirs][:max_entries]
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')

    entries = []
    for source_dir in progress:
        progress.set_description(f'{source_dir:{100}}')

        chapter_id = basename(source_dir)
        speaker_id = basename(abspath(join(source_dir, pardir)))

        if chapter_id not in chapter_meta:
            print(f'WARNING: no chapter information for chapter {chapter_id}. Skipping corpus entry...')
            break

        book_id = chapter_meta[chapter_id]['book_id']
        if not book_id:
            print(f'WARNING: no book text available for chapter {chapter_id}. Skipping corpus entry...')
            break

        segments_file = find_file_by_suffix(source_dir, f'{speaker_id}-{chapter_id}.seg.txt')
        if not segments_file:
            print(f'no segmentation found at {segments_file}. Skipping corpus entry...')
            break

        transcript_file = find_file_by_suffix(source_dir, f'{speaker_id}-{chapter_id}.trans.txt')
        if not transcript_file:
            log.warn(f'no transcript found at {transcript_file}. Skipping corpus entry...')
            break

        mp3_file = find_file_by_suffix(source_dir, f'{chapter_id}.mp3')
        if not mp3_file:
            log.warn(f'no MP3 file found at {mp3_file}. Skipping corpus entry...')
            break

        segments = create_segments(segments_file, transcript_file)

        # crop/resample audio
        wav_file = join(target_dir, basename(splitext(mp3_file)[0] + ".wav"))
        if not exists(wav_file) or args.overwrite:
            crop_and_resample(mp3_file, wav_file, segments)

        # write full transcript
        with open(join(target_dir, f'{chapter_id}.txt'), 'w') as f:
            book_text = normalize(book_texts[book_id], 'en')
            first_transcript = segments[0].transcript
            if first_transcript in book_text:
                text_start = book_text.index(first_transcript)
            else:
                text_start = 0
                # try to find the first transcript by searching the maximum substring from the left
                for i in range(1, len(first_transcript) - 1):
                    if first_transcript[:i] not in book_text:
                        text_start = book_text.index(first_transcript[:i - 1])
                        break

            last_transcript = segments[-1].transcript
            if last_transcript in book_text:
                text_end = book_text.index(last_transcript) + len(last_transcript)
            else:
                # try to find last transcript by searching maximum substring from the right
                text_end = len(book_text) - 1
                for i in range(1, len(last_transcript) - 1):
                    if last_transcript[-i:] not in book_text:
                        text_end = book_text.index(last_transcript[-i + 1:]) + i - 1
                        break

            f.write(book_text[text_start:text_end])

        # Create corpus entry
        subset = chapter_meta[chapter_id]['subset']
        wav_name = basename(wav_file)
        corpus_entry = CorpusEntry(subset, 'en', wav_name, segments)
        entries.append(corpus_entry)

    return entries


def collect_chapter_meta(chapters_file):
    chapter_meta = {'unknown': 'unknown chapter'}

    line_pattern = re.compile("(?P<chapter>\d+)\s*\|.*\|.*\|\s*(?P<subset>.*?)\s*\|.*\|\s*(?P<book>\d+)\s*\|.*\|.*")

    with open(chapters_file) as f:
        for line in (line for line in f.readlines() if not line.startswith(';')):
            result = re.search(line_pattern, line)
            subset = result.group('subset')
            if '-clean-' in subset:
                chapter_id = result.group('chapter')
                book_id = result.group('book')
                chapter_meta[chapter_id] = {'subset': subset, 'book_id': book_id}

    return chapter_meta


def collect_book_texts(books_root):
    book_texts = {}
    invalid_encodings = []
    for root, files in tqdm([(root, files) for root, subdirs, files in walk(books_root)
                             if not subdirs and len(files) == 1], unit='books'):
        book_file = join(root, files[0])
        book_id = basename(root)

        encoding = 'ascii' if 'ascii' in book_file else 'utf-8'
        with open(book_file, 'r', encoding=encoding) as f:
            try:
                book_texts[book_id] = f.read().strip()
            except UnicodeDecodeError as e:
                invalid_encodings.append((book_id, book_file, encoding, e.start, e.end))

    if invalid_encodings:
        print(f'could not decode the following {len(invalid_encodings)} books because of decoding errors:')
        print(tabulate(invalid_encodings, headers=['id', 'file', 'encoding', 'start', 'end']))
        print('trying to fix those files by using Latin-1 encoding and removing invalid HTML markup')

        for book_id, book_file, encoding, start, end in invalid_encodings:
            with open(book_file, 'r', encoding='latin-1') as f:
                book_text = f.read()
                if '<pre>' in book_text:
                    book_text = book_text[:book_text.index('<pre>') + 5]
                if '</pre>' in book_text:
                    book_text = book_text[:book_text.index('</pre>')]
            with open(book_file, 'w', encoding='ascii') as f:
                f.write(unidecode(book_text))

    return book_texts


def create_segments(segments_file, transcript_file):
    transcripts = {}
    with open(transcript_file, 'r') as f_transcript:
        for line in f_transcript.readlines():
            segment_id, transcript = line.split(' ', 1)
            transcripts[segment_id] = transcript.replace('\n', '')

    line_pattern = re.compile('(?P<segment_id>.*)\s(?P<segment_start>.*)\s(?P<segment_end>.*)\n')

    segments = []
    with open(segments_file, 'r') as f_segments:
        lines = f_segments.readlines()
        for i, line in enumerate(lines):
            result = re.search(line_pattern, line)
            if result:
                segment_id = result.group('segment_id')
                start_frame = seconds_to_frame(result.group('segment_start'))
                end_frame = seconds_to_frame(result.group('segment_end'))
                transcript = normalize(transcripts[segment_id], 'en') if segment_id in transcripts else ''

                segment = Segment(start_frame=start_frame, end_frame=end_frame, transcript=transcript, language='en')
                segments.append(segment)
    return segments


if __name__ == '__main__':
    main()
