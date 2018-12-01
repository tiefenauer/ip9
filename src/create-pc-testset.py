from glob import glob
from os import makedirs
from os.path import basename, dirname, join, exists

from pydub.utils import mediainfo
from tqdm import tqdm

from create_rl_corpus import collect_segmentation
from util.audio_util import resample, resample_frame

source_dir = '/media/daniel/Data/corpus/podclub-raw/PodClubDaten/Deutsch'
target_dir = '/media/daniel/IP9/corpora/podclub-test'

for audio_src in tqdm(glob(source_dir + '/**/*.wav', recursive=True)):
    base_dir = dirname(audio_src)
    entry_id = basename(base_dir)
    if not exists(target_dir):
        makedirs(target_dir)

    txt_src = next(iter(glob(base_dir + '/*.txt')))
    txt_dst = join(target_dir, f'{entry_id}.txt')
    with open(txt_src, 'r', encoding='utf-8') as fin, open(txt_dst, 'w') as fout:
        # remove first line (title) and empty lines
        lines = [line for line in fin.readlines()[1:] if len(line.strip()) > 0]
        # remove last lines with glossary
        ix = lines.index(next(iter(line for line in lines if line.startswith('['))))
        lines = lines[:ix]
        transcript = '\n'.join([line.strip() for line in lines])
        fout.write(transcript)

    segmentation_file = next(iter(glob(base_dir + '/*- Segmentation.xml')))
    segments = [s for s in collect_segmentation(segmentation_file) if s['class'] == 'Speech']

    audio_dst = join(target_dir, f'{entry_id}.wav')
    if not exists(audio_dst):
        src_rate = int(mediainfo(audio_src)['sample_rate'])
        crop_start = resample_frame(segments[0]['start_frame'], src_rate=src_rate)
        crop_end = resample_frame(segments[-1]['end_frame'], src_rate=src_rate)
        resample(audio_src, audio_dst, crop_start=crop_start, crop_end=crop_end)
