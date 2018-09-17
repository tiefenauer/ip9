import argparse
from os import listdir, makedirs
from os.path import join, exists, getsize
from shutil import rmtree

import pandas
import soundfile as sf
from tqdm import tqdm

from util.corpus_util import get_corpus
from util.log_util import create_args_str

parser = argparse.ArgumentParser(description="""Export speech segments of corpus to CSV and split audio files""")
parser.add_argument('-id', type=str, help='target-ID for processed files')
parser.add_argument('-c', '--corpus', type=str, choices=['rl', 'ls'], help='corpus to export')
parser.add_argument('-l', '--language', type=str, help='language to use')
parser.add_argument('-t', '--target_dir', type=str, help='target directory to save results')
parser.add_argument('-num', '--include_numeric', action='store_true', default=False,
                    help='(optional) whether to include transcripts with numeric chars (default: False)')
parser.add_argument('-m', '--max', nargs='?', type=int, default=None,
                    help='(optional) maximum number of speech segments to process')
parser.add_argument('-f', '--force', nargs='?', type=bool, action='store_true',
                    help='(optional) force override existing files. Default: False')
args = parser.parse_args()


def main():
    print(create_args_str(args))

    target_dir = join(args.target_dir, args.id)

    if not exists(target_dir):
        print(f'target directory {target_dir} does not exist. Creating...')
        makedirs(target_dir)

    if args.force and listdir(target_dir):
        override = input(f"""WARNING: target directory {target_dir} already exists. Override? 
        (this will remove all files in {target_dir}!!!) (Y/n)        
        """)
        if override.lower() in ['', 'y']:
            rmtree(target_dir)
            makedirs(target_dir)

    corpus = get_corpus(args.corpus)(languages=args.language)
    corpus.summary()

    print(f'processing {corpus.name} corpus and saving split segments in {target_dir}')
    extract_speech_segments(args.id, corpus, target_dir)
    print(f'done! All files are in {target_dir}')


def extract_speech_segments(corpus_id, corpus, target_dir):
    train_set, dev_set, test_set = corpus.train_dev_test_split(include_numeric=args.include_numeric)

    print(f'processing training segments')
    process_subset('train', train_set, corpus_id, target_dir)

    print(f'processing validation segments')
    process_subset('dev', dev_set, corpus_id, target_dir)

    print(f'processing validation segments')
    process_subset('test', test_set, corpus_id, target_dir)


def process_subset(subset_id, subset, corpus_id, target_dir):
    df = split_speech_segments(subset, corpus_id, subset_id, target_dir)

    csv_path = join(target_dir, f'{corpus_id}-{subset_id}.csv')
    print(f'saving metadata in {csv_path}')
    df.to_csv(csv_path, index=False)


def split_speech_segments(subset, corpus_id, subset_id, target_dir):
    files = []

    progress = tqdm(subset, unit=' speech segments')
    for i, segment in enumerate(progress):
        segment_id = f'{corpus_id}-{subset_id}-{i:0=3d}'
        wav_path = join(target_dir, f'{segment_id}.wav')
        txt_path = join(target_dir, f'{segment_id}.txt')

        if not exists(wav_path) or not getsize(wav_path):
            progress.set_description(wav_path)
            sf.write(wav_path, segment.audio, segment.rate, subtype='PCM_16')

        if not exists(txt_path) or not getsize(txt_path):
            with open(txt_path, 'w') as f:
                transcript = f'{segment.start_frame} {segment.end_frame} {segment.text}'
                f.write(transcript)

        files.append((wav_path, getsize(wav_path), segment.text))

    return pandas.DataFrame(data=files, columns=['wav_filename', 'wav_filesize', 'transcript'])


def get_segment_id(segment):
    corpus_entry = segment.corpus_entry
    ix = corpus_entry.speech_segments.index(segment)
    return f'{corpus_entry.id}-{ix:0=3d}'


if __name__ == '__main__':
    main()
